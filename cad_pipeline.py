"""
CAD Prediction Pipeline — YOLO ECG → XGBoost Integration
=========================================================
Transforms YOLO object‑detection output (bounding boxes, class labels,
confidence scores) from ECG images into the structured feature vector
expected by the trained XGBoost coronary‑artery‑disease classifier,
merges it with patient clinical data, and returns a risk probability.

Usage
-----
    from cad_pipeline import CADPredictor

    predictor = CADPredictor("xgb_cad_model.joblib")

    yolo_output = [
        {"class": "ST Elevation", "confidence": 0.87, "bbox": [100, 200, 50, 30]},
        {"class": "Q Wave",       "confidence": 0.72, "bbox": [300, 180, 40, 25]},
    ]

    patient_data = {
        "Age": 62, "Sex": "Male", "BMI": 28.5, "HTN": 1, "DM": 1,
        "BP": 150, "PR": 80, "FBS": 180, "EF-TTE": 40, ...
    }

    result = predictor.predict(yolo_output, patient_data)
    print(result["probability"])   # e.g. 0.83
    print(result["risk_level"])    # "HIGH"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Exact feature columns from CAD.csv (order matters — must match training)
FEATURE_COLUMNS: list[str] = [
    "Age", "Weight", "Length", "Sex", "BMI",
    "DM", "HTN", "Current Smoker", "EX-Smoker", "FH",
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
    "CHF", "DLP", "BP", "PR", "Edema",
    "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur",
    "Diastolic Murmur", "Typical Chest Pain", "Dyspnea",
    "Function Class", "Atypical", "Nonanginal", "Exertional CP",
    "LowTH Ang", "Q Wave", "St Elevation", "St Depression",
    "Tinversion", "LVH", "Poor R Progression",
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB",
    "K", "Na", "WBC", "Lymph", "Neut", "PLT",
    "EF-TTE", "Region RWMA", "VHD",
]

# YOLO class label  →  CAD.csv column name
# Handles casing / spacing differences between YOLO output and CSV schema
YOLO_CLASS_TO_COLUMN: dict[str, str] = {
    # ---- Actual YOLO model class names (from data.yaml) ----
    "LVH":                  "LVH",
    "Pathological Q Wave":  "Q Wave",
    "Poor R Progression":   "Poor R Progression",
    "ST Depression":        "St Depression",
    "T inversion":          "Tinversion",
    # ---- Alternate spellings / casing (backward compatible) ----
    "Q Wave":               "Q Wave",
    "ST Elevation":         "St Elevation",
    "St Elevation":         "St Elevation",
    "St Depression":        "St Depression",
    "T Inversion":          "Tinversion",
    "Tinversion":           "Tinversion",
    # ---- Lowercase fallbacks ----
    "lvh":                  "LVH",
    "pathological q wave":  "Q Wave",
    "q wave":               "Q Wave",
    "st elevation":         "St Elevation",
    "st depression":        "St Depression",
    "t inversion":          "Tinversion",
    "poor r progression":   "Poor R Progression",
}

# The 6 ECG columns populated by YOLO (subset of FEATURE_COLUMNS)
ECG_COLUMNS: list[str] = [
    "Q Wave", "St Elevation", "St Depression",
    "Tinversion", "LVH", "Poor R Progression",
]


@dataclass(frozen=True)
class CADPipelineConfig:
    """Immutable configuration for the CAD prediction pipeline."""

    model_path: str = "xgb_cad_model.joblib"

    # Minimum YOLO confidence to count a detection as positive (binary=1).
    # Lower threshold → higher sensitivity (fewer missed abnormalities).
    # Recommended range for medical use: 0.20–0.35.
    confidence_threshold: float = 0.25

    # Risk‑level probability cut‑offs (for interpretive labelling only)
    low_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.60


# ═══════════════════════════════════════════════════════════════════════
# 2. YOLO TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class YOLOTransformResult:
    """Holds both binary features and confidence metadata."""

    binary_features: dict[str, int]        # column → 0 or 1
    max_confidences: dict[str, float]      # column → highest confidence
    detection_counts: dict[str, int]       # column → number of detections
    unrecognised_classes: list[str]         # class labels not in mapping


def transform_yolo_output(
    yolo_detections: list[dict[str, Any]],
    config: CADPipelineConfig | None = None,
) -> YOLOTransformResult:
    """
    Convert raw YOLO detections into structured ECG features.

    Parameters
    ----------
    yolo_detections : list[dict]
        Each dict must contain at least:
          - "class" (str): detected abnormality label
          - "confidence" (float): detection confidence ∈ [0, 1]
        Optional keys like "bbox" are ignored.
    config : CADPipelineConfig, optional
        Pipeline configuration. Uses defaults if not provided.

    Returns
    -------
    YOLOTransformResult
        Contains binary features, confidence metadata, and any
        unrecognised class labels.
    """
    if config is None:
        config = CADPipelineConfig()

    # Initialise all ECG columns to 0 / 0.0
    binary_features:   dict[str, int]   = {col: 0   for col in ECG_COLUMNS}
    max_confidences:   dict[str, float] = {col: 0.0 for col in ECG_COLUMNS}
    detection_counts:  dict[str, int]   = {col: 0   for col in ECG_COLUMNS}
    unrecognised: list[str] = []

    for det in yolo_detections:
        class_label = str(det.get("class", "")).strip()
        confidence  = float(det.get("confidence", 0.0))

        # Resolve column name (try exact match, then lowercase)
        column = YOLO_CLASS_TO_COLUMN.get(class_label)
        if column is None:
            column = YOLO_CLASS_TO_COLUMN.get(class_label.lower())
        if column is None:
            unrecognised.append(class_label)
            logger.warning("Unrecognised YOLO class: '%s'", class_label)
            continue

        # Aggregate: keep max confidence across all detections of same class
        detection_counts[column] += 1
        max_confidences[column] = max(max_confidences[column], confidence)

        # Binary: 1 if any detection meets threshold
        if confidence >= config.confidence_threshold:
            binary_features[column] = 1

    if unrecognised:
        logger.warning(
            "Unrecognised YOLO classes (ignored): %s", unrecognised
        )

    return YOLOTransformResult(
        binary_features=binary_features,
        max_confidences=max_confidences,
        detection_counts=detection_counts,
        unrecognised_classes=unrecognised,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE VECTOR BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_feature_vector(
    yolo_output: list[dict[str, Any]],
    patient_data: dict[str, Any],
    config: CADPipelineConfig | None = None,
) -> tuple[pd.DataFrame, YOLOTransformResult]:
    """
    Merge YOLO ECG detections with patient clinical data into a single
    DataFrame row that exactly matches the XGBoost training schema.

    Parameters
    ----------
    yolo_output : list[dict]
        Raw YOLO detections (see `transform_yolo_output`).
    patient_data : dict
        Patient clinical features keyed by CSV column names.
        Missing keys are set to NaN (handled by the preprocessor imputer).
    config : CADPipelineConfig, optional

    Returns
    -------
    (features_df, yolo_result)
        - features_df: 1‑row DataFrame with columns in training order
        - yolo_result: YOLOTransformResult for logging / interpretability
    """
    if config is None:
        config = CADPipelineConfig()

    # --- Step 1: transform YOLO detections ---
    yolo_result = transform_yolo_output(yolo_output, config)

    # --- Step 2: build row from patient data ---
    row: dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col in yolo_result.binary_features:
            # ECG feature — use YOLO‑derived binary value
            row[col] = yolo_result.binary_features[col]
        elif col in patient_data:
            row[col] = patient_data[col]
        else:
            # Missing → NaN (the preprocessor's imputer will handle it)
            row[col] = np.nan

    # --- Step 3: allow patient_data to override YOLO ECG features ---
    # If the clinician explicitly supplies an ECG column in patient_data,
    # their value takes precedence (manual override).
    for col in ECG_COLUMNS:
        if col in patient_data and patient_data[col] is not None:
            logger.info(
                "Patient data overrides YOLO for '%s': %s → %s",
                col, row[col], patient_data[col],
            )
            row[col] = patient_data[col]

    # --- Step 4: create DataFrame with exact column order ---
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # Log any columns that ended up as NaN for traceability
    missing = [c for c in FEATURE_COLUMNS if pd.isna(df[c].iloc[0])]
    if missing:
        logger.info("Missing features (will be imputed): %s", missing)

    return df, yolo_result


# ═══════════════════════════════════════════════════════════════════════
# 4. PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def _classify_risk(probability: float, config: CADPipelineConfig) -> str:
    """Map probability to a human‑readable risk level."""
    if probability >= config.high_risk_threshold:
        return "HIGH"
    elif probability >= config.low_risk_threshold:
        return "MODERATE"
    return "LOW"


def predict_cad(
    yolo_output: list[dict[str, Any]],
    patient_data: dict[str, Any],
    model_path: str | None = None,
    config: CADPipelineConfig | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    """
    End‑to‑end CAD prediction from YOLO detections + clinical data.

    Parameters
    ----------
    yolo_output : list[dict]
        Raw YOLO detection list.
    patient_data : dict
        Patient clinical features.
    model_path : str, optional
        Path to the saved joblib model bundle. Defaults to config value.
    config : CADPipelineConfig, optional
    return_details : bool
        If True, include the full feature vector and YOLO metadata
        in the result (useful for debugging / audit trails).

    Returns
    -------
    dict with keys:
        - probability (float): CAD probability ∈ [0, 1]
        - prediction (int): 1 = CAD predicted, 0 = Normal
        - risk_level (str): "LOW", "MODERATE", or "HIGH"
        - ecg_detections (dict): YOLO binary features used
        - confidence_metadata (dict): max confidence per ECG class
      If return_details=True, also includes:
        - feature_vector (dict): full feature row
        - detection_counts (dict): number of detections per class
    """
    if config is None:
        config = CADPipelineConfig()
    if model_path is None:
        model_path = config.model_path

    # Build feature vector
    features_df, yolo_result = build_feature_vector(
        yolo_output, patient_data, config
    )

    # Load model bundle
    bundle = joblib.load(model_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    # Preprocess and predict
    X_processed = preprocessor.transform(features_df)
    probability = float(model.predict_proba(X_processed)[0][1])
    prediction = int(probability >= 0.5)

    result: dict[str, Any] = {
        "probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": _classify_risk(probability, config),
        "ecg_detections": yolo_result.binary_features,
        "confidence_metadata": {
            k: round(v, 4) for k, v in yolo_result.max_confidences.items()
        },
    }

    if return_details:
        result["feature_vector"] = features_df.iloc[0].to_dict()
        result["detection_counts"] = yolo_result.detection_counts
        result["unrecognised_classes"] = yolo_result.unrecognised_classes

    return result


# ═══════════════════════════════════════════════════════════════════════
# 5. PREDICTOR CLASS (for server / repeated use)
# ═══════════════════════════════════════════════════════════════════════

class CADPredictor:
    """
    Stateful predictor that loads the model once and reuses it.
    Designed for integration into a backend server (FastAPI, Flask, etc.).

    Example
    -------
        predictor = CADPredictor("xgb_cad_model.joblib")
        result = predictor.predict(yolo_output, patient_data)
    """

    def __init__(
        self,
        model_path: str = "xgb_cad_model.joblib",
        config: CADPipelineConfig | None = None,
    ):
        self.config = config or CADPipelineConfig(model_path=model_path)
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load and validate the model bundle."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path.resolve()}"
            )

        bundle = joblib.load(path)

        if not isinstance(bundle, dict):
            raise ValueError("Model bundle must be a dict")
        if "preprocessor" not in bundle or "model" not in bundle:
            raise ValueError(
                "Model bundle must contain 'preprocessor' and 'model' keys"
            )

        self._preprocessor = bundle["preprocessor"]
        self._model = bundle["model"]
        logger.info("Model loaded from %s", path)

    def predict(
        self,
        yolo_output: list[dict[str, Any]],
        patient_data: dict[str, Any],
        return_details: bool = False,
    ) -> dict[str, Any]:
        """
        Run CAD prediction using the pre‑loaded model.

        Parameters and return value are identical to `predict_cad()`
        except the model is not re‑loaded on each call.
        """
        features_df, yolo_result = build_feature_vector(
            yolo_output, patient_data, self.config
        )

        X_processed = self._preprocessor.transform(features_df)
        probability = float(self._model.predict_proba(X_processed)[0][1])
        prediction = int(probability >= 0.5)

        result: dict[str, Any] = {
            "probability": round(probability, 4),
            "prediction": prediction,
            "risk_level": _classify_risk(probability, self.config),
            "ecg_detections": yolo_result.binary_features,
            "confidence_metadata": {
                k: round(v, 4)
                for k, v in yolo_result.max_confidences.items()
            },
        }

        if return_details:
            result["feature_vector"] = features_df.iloc[0].to_dict()
            result["detection_counts"] = yolo_result.detection_counts
            result["unrecognised_classes"] = yolo_result.unrecognised_classes

        return result


# ═══════════════════════════════════════════════════════════════════════
# 6. CLI DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ---------- Sample YOLO output ----------
    sample_yolo = [
        {"class": "ST Elevation", "confidence": 0.87, "bbox": [100, 200, 50, 30]},
        {"class": "Q Wave",       "confidence": 0.72, "bbox": [300, 180, 40, 25]},
        {"class": "T Inversion",  "confidence": 0.15, "bbox": [450, 190, 35, 20]},
        {"class": "LVH",          "confidence": 0.91, "bbox": [500, 210, 45, 28]},
    ]

    # ---------- Sample patient data ----------
    sample_patient = {
        "Age": 62, "Weight": 80, "Length": 170, "Sex": "Male",
        "BMI": 27.7, "DM": 1, "HTN": 1, "Current Smoker": 0,
        "EX-Smoker": 0, "FH": 1, "Obesity": "Y", "CRF": "N",
        "CVA": "N", "Airway disease": "N", "Thyroid Disease": "N",
        "CHF": "N", "DLP": "Y", "BP": 150, "PR": 80,
        "Edema": 0, "Weak Peripheral Pulse": "N", "Lung rales": "N",
        "Systolic Murmur": "N", "Diastolic Murmur": "N",
        "Typical Chest Pain": 1, "Dyspnea": "Y",
        "Function Class": 2, "Atypical": "N", "Nonanginal": "N",
        "Exertional CP": 0, "LowTH Ang": 0,
        "FBS": 180, "CR": 1.2, "TG": 200, "LDL": 140, "HDL": 35,
        "BUN": 22, "ESR": 25, "HB": 13.5, "K": 4.5, "Na": 140,
        "WBC": 8000, "Lymph": 30, "Neut": 60, "PLT": 250,
        "EF-TTE": 40, "Region RWMA": 2, "VHD": "mild",
    }

    print("=" * 60)
    print("  CARDIONYX — CAD PREDICTION PIPELINE DEMO")
    print("=" * 60)

    # Functional API
    result = predict_cad(
        sample_yolo, sample_patient, return_details=True
    )

    print(f"\n🫀 CAD Probability : {result['probability'] * 100:.1f}%")
    print(f"📊 Risk Level      : {result['risk_level']}")
    print(f"🔍 Prediction      : {'CAD' if result['prediction'] else 'Normal'}")

    print("\n📋 ECG Detections (from YOLO):")
    for feat, val in result["ecg_detections"].items():
        conf = result["confidence_metadata"][feat]
        status = "✔ DETECTED" if val else "✘ not detected"
        print(f"   {feat:<25s} → {status}  (conf: {conf:.2f})")

    print("\n" + "=" * 60)
