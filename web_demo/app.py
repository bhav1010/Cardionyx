"""
Cardionyx Web Demo — Flask Backend
====================================
Serves the prediction form, accepts POST /predict with:
  - Clinical form fields (multipart/form-data)
  - Optional ECG image (runs YOLO if provided)

Run:
    python app.py
    # or
    flask --app web_demo/app.py run
"""

from __future__ import annotations

import os
import sys
import logging
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory

# ── Add project root to path so cad_pipeline can be imported ──────────────
ROOT = Path(__file__).resolve().parent.parent            # …/Cardionyx/
sys.path.insert(0, str(ROOT))

from cad_pipeline import (
    CADPredictor,
    CADPipelineConfig,
    ECG_COLUMNS,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cardionyx_web")

# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024   # 12 MB upload limit

# ── Model paths ───────────────────────────────────────────────────────────
XGB_MODEL_PATH  = ROOT / "xgb_cad_model.joblib"
YOLO_MODEL_PATH = ROOT / "training_runs" / "yolo_run5" / "weights" / "best.pt"
YOLO_CONF_THR   = 0.25

# ── Load XGBoost model at startup ─────────────────────────────────────────
_predictor: CADPredictor | None = None

def get_predictor() -> CADPredictor:
    global _predictor
    if _predictor is None:
        config = CADPipelineConfig(
            model_path=str(XGB_MODEL_PATH),
            confidence_threshold=YOLO_CONF_THR,
        )
        _predictor = CADPredictor(str(XGB_MODEL_PATH), config=config)
        logger.info("XGBoost model loaded from %s", XGB_MODEL_PATH)
    return _predictor


# ── Field parsing helpers ─────────────────────────────────────────────────

def _float_or_none(val: str):
    """Return float if parseable, else None (→ model imputer handles it)."""
    try:
        return float(val) if val.strip() else None
    except (ValueError, AttributeError):
        return None


def _int_or_default(val: str, default: int) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def parse_patient_data(form) -> dict:
    """
    Convert raw form fields into the dict expected by cad_pipeline.
    All optional fields default to None/NaN to let the model impute.
    """
    import numpy as np

    def flt(key):    return _float_or_none(form.get(key, ''))
    def yn(key):     return form.get(key, 'N')
    def binary(key): return _int_or_default(form.get(key, '0'), 0)

    # Demographics
    age    = flt('Age')
    weight = flt('Weight')
    height = flt('Length')  # cm
    sex    = form.get('Sex', '')
    bmi_raw = flt('BMI')
    if bmi_raw is None and weight and height and height > 0:
        bmi_raw = round(weight / ((height / 100) ** 2), 2)

    data = {
        # Demographics
        "Age":    age,
        "Weight": weight,
        "Length": height,
        "Sex":    sex if sex else np.nan,
        "BMI":    bmi_raw if bmi_raw is not None else np.nan,

        # Medical history
        "DM":              binary('DM'),
        "HTN":             binary('HTN'),
        "Current Smoker":  binary('Current Smoker'),
        "EX-Smoker":       binary('EX-Smoker'),
        "FH":              binary('FH'),
        "Obesity":         yn('Obesity'),
        "CRF":             yn('CRF'),
        "CVA":             yn('CVA'),
        "Airway disease":  yn('Airway disease'),
        "Thyroid Disease": yn('Thyroid Disease'),
        "CHF":             yn('CHF'),
        "DLP":             yn('DLP'),

        # Vitals
        "BP":                    flt('BP') or np.nan,
        "PR":                    flt('PR') or np.nan,
        "Edema":                 binary('Edema'),
        "Weak Peripheral Pulse": yn('Weak Peripheral Pulse'),
        "Lung rales":            yn('Lung rales'),
        "Systolic Murmur":       yn('Systolic Murmur'),
        "Diastolic Murmur":      yn('Diastolic Murmur'),

        # Symptoms
        "Typical Chest Pain": binary('Typical Chest Pain'),
        "Dyspnea":            yn('Dyspnea'),
        "Function Class":     _int_or_default(form.get('Function Class', '0'), 0),
        "Atypical":           yn('Atypical'),
        "Nonanginal":         yn('Nonanginal'),
        "Exertional CP":      binary('Exertional CP'),
        "LowTH Ang":          binary('LowTH Ang'),

        # Labs (all optional)
        "FBS":  flt('FBS')  or np.nan,
        "CR":   flt('CR')   or np.nan,
        "TG":   flt('TG')   or np.nan,
        "LDL":  flt('LDL')  or np.nan,
        "HDL":  flt('HDL')  or np.nan,
        "BUN":  flt('BUN')  or np.nan,
        "ESR":  flt('ESR')  or np.nan,
        "HB":   flt('HB')   or np.nan,
        "K":    flt('K')    or np.nan,
        "Na":   flt('Na')   or np.nan,
        "WBC":  flt('WBC')  or np.nan,
        "Lymph":flt('Lymph')or np.nan,
        "Neut": flt('Neut') or np.nan,
        "PLT":  flt('PLT')  or np.nan,

        # Heart tests
        "EF-TTE":     flt('EF-TTE') or np.nan,
        "Region RWMA":_int_or_default(form.get('Region RWMA', '0'), 0),
        "VHD":        form.get('VHD', 'N'),
    }

    return data


def run_yolo(image_path: str) -> list[dict]:
    """Run YOLO inference on an ECG image; returns list of detection dicts."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics not installed — skipping YOLO inference")
        return []

    yolo_path = YOLO_MODEL_PATH
    if not yolo_path.exists():
        logger.warning("YOLO model not found at %s — skipping", yolo_path)
        return []

    model   = YOLO(str(yolo_path))
    results = model(image_path, verbose=False, conf=YOLO_CONF_THR)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = r.names[cls_id]
            bbox       = box.xyxy[0].tolist()
            detections.append({
                "class":      class_name,
                "confidence": round(conf, 4),
                "bbox":       [round(c, 1) for c in bbox],
            })

    return detections


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        predictor = get_predictor()
    except Exception as exc:
        logger.error("Model load error: %s", exc)
        return jsonify({"error": f"Failed to load model: {exc}"}), 500

    # ---- Parse patient data ----
    patient_data = parse_patient_data(request.form)

    # ---- Handle optional ECG image ----
    yolo_detections: list[dict] = []
    ecg_file = request.files.get("ecg_image")

    if ecg_file and ecg_file.filename:
        # Save to a temp file and run YOLO
        suffix = Path(ecg_file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            ecg_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            yolo_detections = run_yolo(tmp_path)
            logger.info(
                "YOLO found %d detection(s) for '%s'",
                len(yolo_detections), ecg_file.filename,
            )
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ---- Run CAD prediction ----
    try:
        result = predictor.predict(yolo_detections, patient_data, return_details=True)
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    # ---- Build response ----
    return jsonify({
        "probability":           result["probability"],
        "prediction":            result["prediction"],
        "risk_level":            result["risk_level"],
        "ecg_detections":        result["ecg_detections"],
        "confidence_metadata":   result["confidence_metadata"],
        "ecg_used":              bool(ecg_file and ecg_file.filename),
        "yolo_detection_count":  len(yolo_detections),
    })


# ── Dev server ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Cardionyx Web Demo")
    print("  ═══════════════════════════════")
    print(f"  XGBoost model : {XGB_MODEL_PATH}")
    print(f"  YOLO model    : {YOLO_MODEL_PATH}")
    print("  Open browser  : http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
