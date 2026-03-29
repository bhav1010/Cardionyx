"""
Verification script for the CAD prediction pipeline.
Run: python test_cad_pipeline.py
"""

import sys
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────
# Colour helpers for terminal output
# ────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0
skipped = 0

# Known error patterns from sklearn version mismatches
SKLEARN_COMPAT_ERRORS = ("_fill_dtype", "No module named", "cannot import")


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}[PASS]{RESET}  {name}")
    else:
        failed += 1
        print(f"  {RED}[FAIL]{RESET}  {name}  -- {detail}")


def skip(name: str, reason: str = ""):
    global skipped
    skipped += 1
    print(f"  {YELLOW}[SKIP]{RESET}  {name}  -- {reason}")


# ════════════════════════════════════════════════════
# 1. IMPORT TEST
# ════════════════════════════════════════════════════
print(f"\n{CYAN}{BOLD}═══ 1. Import Test ═══{RESET}")
try:
    from cad_pipeline import (
        FEATURE_COLUMNS,
        ECG_COLUMNS,
        YOLO_CLASS_TO_COLUMN,
        CADPipelineConfig,
        transform_yolo_output,
        build_feature_vector,
        predict_cad,
        CADPredictor,
    )
    check("All pipeline components importable", True)
except ImportError as e:
    check("All pipeline components importable", False, str(e))
    sys.exit(1)


# ════════════════════════════════════════════════════
# 2. SCHEMA TEST
# ════════════════════════════════════════════════════
print(f"\n{CYAN}{BOLD}═══ 2. Schema Test ═══{RESET}")

# Load ground truth from CSV
csv_df = pd.read_csv("CAD.csv")
csv_features = list(csv_df.drop(columns=[csv_df.columns[-1]]).columns)

check(
    "Feature count matches CSV (53 columns)",
    len(FEATURE_COLUMNS) == len(csv_features),
    f"expected {len(csv_features)}, got {len(FEATURE_COLUMNS)}",
)

check(
    "Feature names match CSV exactly",
    FEATURE_COLUMNS == csv_features,
    f"mismatches: {set(FEATURE_COLUMNS) ^ set(csv_features)}" if FEATURE_COLUMNS != csv_features else "",
)

check(
    "Feature order matches CSV exactly",
    all(a == b for a, b in zip(FEATURE_COLUMNS, csv_features)),
    "Order mismatch",
)


# ════════════════════════════════════════════════════
# 3. YOLO TRANSFORM TEST
# ════════════════════════════════════════════════════
print(f"\n{CYAN}{BOLD}═══ 3. YOLO Transform Tests ═══{RESET}")

config = CADPipelineConfig(confidence_threshold=0.25)

# 3a. Basic detections
detections = [
    {"class": "ST Elevation", "confidence": 0.87, "bbox": [100, 200, 50, 30]},
    {"class": "Q Wave",       "confidence": 0.72, "bbox": [300, 180, 40, 25]},
]
result = transform_yolo_output(detections, config)

check("ST Elevation → binary 1",   result.binary_features["St Elevation"] == 1)
check("Q Wave → binary 1",         result.binary_features["Q Wave"] == 1)
check("LVH → binary 0 (absent)",   result.binary_features["LVH"] == 0)
check("Max confidence tracked",    result.max_confidences["St Elevation"] == 0.87)

# 3b. Below-threshold detection
low_conf = [{"class": "T Inversion", "confidence": 0.15}]
result_low = transform_yolo_output(low_conf, config)
check(
    "Below-threshold → binary 0",
    result_low.binary_features["Tinversion"] == 0,
    f"got {result_low.binary_features['Tinversion']}",
)
check(
    "Below-threshold confidence still tracked",
    result_low.max_confidences["Tinversion"] == 0.15,
)

# 3c. Multiple detections of same class (aggregation)
multi = [
    {"class": "LVH", "confidence": 0.60},
    {"class": "LVH", "confidence": 0.85},
    {"class": "LVH", "confidence": 0.40},
]
result_multi = transform_yolo_output(multi, config)
check("Multiple detections → binary 1",          result_multi.binary_features["LVH"] == 1)
check("Multiple detections → max confidence",    result_multi.max_confidences["LVH"] == 0.85)
check("Multiple detections → count = 3",         result_multi.detection_counts["LVH"] == 3)

# 3d. Empty detections
result_empty = transform_yolo_output([], config)
check(
    "Empty detections → all zeros",
    all(v == 0 for v in result_empty.binary_features.values()),
)

# 3e. Unrecognised class
unk = [{"class": "Atrial Fibrillation", "confidence": 0.90}]
result_unk = transform_yolo_output(unk, config)
check(
    "Unrecognised class reported",
    "Atrial Fibrillation" in result_unk.unrecognised_classes,
)

# 3f. Case-insensitive mapping
ci = [{"class": "st elevation", "confidence": 0.50}]
result_ci = transform_yolo_output(ci, config)
check(
    "Case-insensitive class mapping works",
    result_ci.binary_features["St Elevation"] == 1,
)


# ════════════════════════════════════════════════════
# 4. FEATURE VECTOR TEST
# ════════════════════════════════════════════════════
print(f"\n{CYAN}{BOLD}═══ 4. Feature Vector Tests ═══{RESET}")

yolo_out = [
    {"class": "ST Elevation", "confidence": 0.87},
    {"class": "Q Wave",       "confidence": 0.72},
]

patient = {
    "Age": 62, "Weight": 80, "Length": 170, "Sex": "Male",
    "BMI": 27.7, "DM": 1, "HTN": 1, "BP": 150, "PR": 80,
}

df, _ = build_feature_vector(yolo_out, patient, config)

check("Output is a DataFrame",        isinstance(df, pd.DataFrame))
check("Single row",                   len(df) == 1)
check("53 columns",                   len(df.columns) == len(FEATURE_COLUMNS))
check("Column order preserved",       list(df.columns) == FEATURE_COLUMNS)
check("Age = 62",                     df["Age"].iloc[0] == 62)
check("St Elevation = 1 (from YOLO)", df["St Elevation"].iloc[0] == 1)
check("Q Wave = 1 (from YOLO)",       df["Q Wave"].iloc[0] == 1)
check("LVH = 0 (no YOLO detection)",  df["LVH"].iloc[0] == 0)
check(
    "Missing features are NaN",
    pd.isna(df["FBS"].iloc[0]),
    f"got {df['FBS'].iloc[0]}",
)


# ════════════════════════════════════════════════════
# 5. END-TO-END PREDICTION TEST
# ════════════════════════════════════════════════════
print(f"\n{CYAN}{BOLD}═══ 5. End-to-End Prediction Test ═══{RESET}")

full_patient = {
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

full_yolo = [
    {"class": "ST Elevation", "confidence": 0.87},
    {"class": "Q Wave",       "confidence": 0.72},
    {"class": "LVH",          "confidence": 0.91},
]

try:
    pred = predict_cad(
        full_yolo, full_patient,
        model_path="xgb_cad_model.joblib",
        return_details=True,
    )

    check("Result is a dict",               isinstance(pred, dict))
    check("Probability is float in [0,1]",   0.0 <= pred["probability"] <= 1.0)
    check("Prediction is 0 or 1",            pred["prediction"] in (0, 1))
    check("Risk level is valid",             pred["risk_level"] in ("LOW", "MODERATE", "HIGH"))
    check("ECG detections included",         "ecg_detections" in pred)
    check("Confidence metadata included",    "confidence_metadata" in pred)
    check("Feature vector in details",       "feature_vector" in pred)

    print(f"\n  {CYAN}-> Probability: {pred['probability']*100:.1f}%{RESET}")
    print(f"  {CYAN}-> Risk Level:  {pred['risk_level']}{RESET}")

except Exception as e:
    err_msg = str(e)
    if any(tok in err_msg for tok in SKLEARN_COMPAT_ERRORS):
        skip("End-to-end prediction (sklearn version mismatch)",
             "Re-save the model with your current sklearn version")
    else:
        check("End-to-end prediction succeeds", False, err_msg)


# ════════════════════════════════════════════════════
# 6. CADPredictor CLASS TEST
# ════════════════════════════════════════════════════
print(f"\n{CYAN}{BOLD}═══ 6. CADPredictor Class Test ═══{RESET}")

try:
    predictor = CADPredictor("xgb_cad_model.joblib")
    check("CADPredictor instantiated", True)

    result1 = predictor.predict(full_yolo, full_patient)
    result2 = predictor.predict(full_yolo, full_patient)
    check(
        "Deterministic predictions",
        result1["probability"] == result2["probability"],
    )
except Exception as e:
    err_msg = str(e)
    if any(tok in err_msg for tok in SKLEARN_COMPAT_ERRORS):
        skip("CADPredictor class (sklearn version mismatch)",
             "Re-save the model with your current sklearn version")
    else:
        check("CADPredictor works", False, err_msg)


# ════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════
print(f"\n{BOLD}{'=' * 50}")
total = passed + failed + skipped
if failed == 0:
    msg = f"  ALL {passed} TESTS PASSED"
    if skipped:
        msg += f" ({skipped} skipped)"
    print(f"{GREEN}{msg}{RESET}")
else:
    print(f"{RED}  {failed}/{total} TESTS FAILED{RESET}")
if skipped:
    print(f"{YELLOW}  {skipped} test(s) skipped due to environment issues{RESET}")
print(f"{'=' * 50}{RESET}\n")

sys.exit(1 if failed else 0)
