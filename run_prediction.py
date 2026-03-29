"""
Cardionyx — Interactive CAD Prediction
=======================================
Accepts patient clinical data via terminal prompts and an ECG image path,
runs YOLO inference on the image, and outputs a CAD risk prediction.

Usage:
    python run_prediction.py
    python run_prediction.py --ecg path/to/ecg_image.jpg
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

from cad_pipeline import (
    CADPredictor,
    CADPipelineConfig,
    FEATURE_COLUMNS,
    ECG_COLUMNS,
)

# ═══════════════════════════════════════════════════════════════════════
# ANSI COLOURS
# ═══════════════════════════════════════════════════════════════════════
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# ═══════════════════════════════════════════════════════════════════════
# DEFAULT MODEL PATHS
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_XGB_MODEL  = "xgb_cad_model.joblib"
DEFAULT_YOLO_MODEL = "training_runs/yolo_run5/weights/best.pt"

# ═══════════════════════════════════════════════════════════════════════
# INPUT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════
# Each entry: (column_name, prompt_label, input_type, default_value)
#   input_type: "float", "int", "binary", "yn", "sex", "vhd", "funcclass"

PATIENT_FIELDS = [
    # ---- Demographics ----
    ("Age",              "Age (years)",                     "float", None),
    ("Weight",           "Weight (kg)",                     "float", None),
    ("Length",           "Height (cm)",                     "float", None),
    ("Sex",             "Sex (Male/Fmale)",                 "sex",   None),
    ("BMI",             "BMI (auto-calc if blank)",         "float", "auto"),
    # ---- Medical History ----
    ("DM",              "Diabetes Mellitus (0/1)",          "binary", 0),
    ("HTN",             "Hypertension (0/1)",               "binary", 0),
    ("Current Smoker",  "Current Smoker (0/1)",             "binary", 0),
    ("EX-Smoker",       "Ex-Smoker (0/1)",                  "binary", 0),
    ("FH",              "Family History of CAD (0/1)",      "binary", 0),
    ("Obesity",         "Obesity (Y/N)",                    "yn",    "N"),
    ("CRF",             "Chronic Renal Failure (Y/N)",      "yn",    "N"),
    ("CVA",             "Cerebrovascular Accident (Y/N)",   "yn",    "N"),
    ("Airway disease",  "Airway Disease (Y/N)",             "yn",    "N"),
    ("Thyroid Disease", "Thyroid Disease (Y/N)",            "yn",    "N"),
    ("CHF",             "Congestive Heart Failure (Y/N)",   "yn",    "N"),
    ("DLP",             "Dyslipidemia (Y/N)",               "yn",    "N"),
    # ---- Vitals & Examination ----
    ("BP",              "Blood Pressure - MAP (mmHg)",      "float", None),
    ("PR",              "Pulse Rate (/min)",                "float", None),
    ("Edema",           "Edema (0/1)",                      "binary", 0),
    ("Weak Peripheral Pulse", "Weak Peripheral Pulse (Y/N)", "yn",  "N"),
    ("Lung rales",      "Lung Rales (Y/N)",                 "yn",    "N"),
    ("Systolic Murmur", "Systolic Murmur (Y/N)",            "yn",    "N"),
    ("Diastolic Murmur","Diastolic Murmur (Y/N)",           "yn",    "N"),
    # ---- Symptoms ----
    ("Typical Chest Pain", "Typical Chest Pain (0/1)",      "binary", 0),
    ("Dyspnea",         "Dyspnea (Y/N)",                    "yn",    "N"),
    ("Function Class",  "NYHA Function Class (0-3)",        "funcclass", 0),
    ("Atypical",        "Atypical Angina (Y/N)",            "yn",    "N"),
    ("Nonanginal",      "Nonanginal Chest Pain (Y/N)",      "yn",    "N"),
    ("Exertional CP",   "Exertional Chest Pain (0/1)",      "binary", 0),
    ("LowTH Ang",       "Low Threshold Angina (0/1)",       "binary", 0),
    # ---- Lab Results ----
    ("FBS",             "Fasting Blood Sugar (mg/dL)",      "float", None),
    ("CR",              "Creatinine (mg/dL)",               "float", None),
    ("TG",              "Triglycerides (mg/dL)",             "float", None),
    ("LDL",             "LDL Cholesterol (mg/dL)",          "float", None),
    ("HDL",             "HDL Cholesterol (mg/dL)",          "float", None),
    ("BUN",             "Blood Urea Nitrogen (mg/dL)",      "float", None),
    ("ESR",             "ESR",                              "float", None),
    ("HB",              "Hemoglobin (g/dL)",                "float", None),
    ("K",               "Potassium (mmol/L)",               "float", None),
    ("Na",              "Sodium (mmol/L)",                   "float", None),
    ("WBC",             "WBC count (/dL)",                  "float", None),
    ("Lymph",           "Lymphocyte %",                     "float", None),
    ("Neut",            "Neutrophil %",                     "float", None),
    ("PLT",             "Platelet count",                   "float", None),
    # ---- Heart Tests ----
    ("EF-TTE",          "Ejection Fraction EF-TTE (%)",     "float", None),
    ("Region RWMA",     "Region RWMA (0-4)",                "int",   0),
    ("VHD",             "Valvular Heart Disease (N/mild/moderate/Severe)", "vhd", "N"),
]


# ═══════════════════════════════════════════════════════════════════════
# INPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════

def prompt_value(label: str, input_type: str, default):
    """Prompt the user for a single value with validation."""
    default_str = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {CYAN}{label}{default_str}{RESET}: ").strip()

        # Use default if blank
        if raw == "" and default is not None:
            return default

        # Skip (leave as NaN)
        if raw == "" and default is None:
            return np.nan

        try:
            if input_type == "float":
                return float(raw)
            elif input_type == "int":
                return int(raw)
            elif input_type == "binary":
                if raw in ("0", "1"):
                    return int(raw)
                print(f"    {YELLOW}Enter 0 or 1{RESET}")
            elif input_type == "yn":
                if raw.upper() in ("Y", "N"):
                    return raw.upper()
                print(f"    {YELLOW}Enter Y or N{RESET}")
            elif input_type == "sex":
                if raw.lower() in ("male", "m"):
                    return "Male"
                elif raw.lower() in ("female", "f", "fmale"):
                    return "Fmale"
                print(f"    {YELLOW}Enter Male or Female{RESET}")
            elif input_type == "vhd":
                valid = {"n": "N", "mild": "mild", "moderate": "moderate",
                         "moderate": "Moderate", "severe": "Severe", "s": "Severe"}
                if raw.lower() in valid:
                    return valid[raw.lower()]
                print(f"    {YELLOW}Enter N, mild, moderate, or Severe{RESET}")
            elif input_type == "funcclass":
                if raw in ("0", "1", "2", "3"):
                    return int(raw)
                print(f"    {YELLOW}Enter 0, 1, 2, or 3{RESET}")
            else:
                return raw
        except ValueError:
            print(f"    {YELLOW}Invalid input, try again{RESET}")


def collect_patient_data() -> dict:
    """Interactively collect all patient clinical data."""
    data = {}

    print(f"\n{BOLD}{WHITE}--- DEMOGRAPHICS ---{RESET}")
    for col, label, itype, default in PATIENT_FIELDS[:5]:
        if col == "BMI":
            val = prompt_value(label, itype, default)
            if val == "auto" and "Weight" in data and "Length" in data:
                w = data["Weight"]
                h = data["Length"] / 100  # cm → m
                if isinstance(w, (int, float)) and isinstance(h, (int, float)) and h > 0:
                    val = round(w / (h ** 2), 2)
                    print(f"    {GREEN}BMI auto-calculated: {val}{RESET}")
                else:
                    val = np.nan
            data[col] = val
        else:
            data[col] = prompt_value(label, itype, default)

    print(f"\n{BOLD}{WHITE}--- MEDICAL HISTORY ---{RESET}")
    for col, label, itype, default in PATIENT_FIELDS[5:17]:
        data[col] = prompt_value(label, itype, default)

    print(f"\n{BOLD}{WHITE}--- VITALS & EXAMINATION ---{RESET}")
    for col, label, itype, default in PATIENT_FIELDS[17:24]:
        data[col] = prompt_value(label, itype, default)

    print(f"\n{BOLD}{WHITE}--- SYMPTOMS ---{RESET}")
    for col, label, itype, default in PATIENT_FIELDS[24:31]:
        data[col] = prompt_value(label, itype, default)

    print(f"\n{BOLD}{WHITE}--- LAB RESULTS ---{RESET}")
    print(f"  {DIM}(press Enter to skip any unknown values){RESET}")
    for col, label, itype, default in PATIENT_FIELDS[31:45]:
        data[col] = prompt_value(label, itype, default)

    print(f"\n{BOLD}{WHITE}--- HEART TESTS ---{RESET}")
    for col, label, itype, default in PATIENT_FIELDS[45:]:
        data[col] = prompt_value(label, itype, default)

    return data


# ═══════════════════════════════════════════════════════════════════════
# YOLO INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def run_yolo_on_image(image_path: str, yolo_model_path: str) -> list[dict]:
    """
    Run YOLO inference on an ECG image and return detections
    in the format expected by cad_pipeline.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"{RED}ultralytics not installed. Install with: pip install ultralytics{RESET}")
        return []

    path = Path(image_path)
    if not path.exists():
        print(f"{RED}Image not found: {path}{RESET}")
        return []

    model_path = Path(yolo_model_path)
    if not model_path.exists():
        print(f"{RED}YOLO model not found: {model_path}{RESET}")
        return []

    print(f"\n{CYAN}Running YOLO inference on: {path.name}...{RESET}")
    model = YOLO(str(model_path))
    results = model(str(path), verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            bbox = box.xyxy[0].tolist()

            detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "bbox": [round(c, 1) for c in bbox],
            })

    return detections


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════

def display_results(result: dict, detections: list[dict]):
    """Pretty-print the CAD prediction results."""
    prob = result["probability"]
    risk = result["risk_level"]

    risk_color = GREEN if risk == "LOW" else YELLOW if risk == "MODERATE" else RED

    print(f"\n{BOLD}{'=' * 56}")
    print(f"          CARDIONYX - CAD PREDICTION RESULT")
    print(f"{'=' * 56}{RESET}")

    print(f"\n  {BOLD}CAD Probability  : {risk_color}{prob * 100:.1f}%{RESET}")
    print(f"  {BOLD}Risk Level       : {risk_color}{risk}{RESET}")
    print(f"  {BOLD}Prediction       : {risk_color}{'CAD' if result['prediction'] else 'Normal'}{RESET}")

    print(f"\n  {WHITE}{BOLD}ECG Findings (from YOLO):{RESET}")
    ecg = result["ecg_detections"]
    conf_meta = result["confidence_metadata"]
    for feat, val in ecg.items():
        conf = conf_meta.get(feat, 0.0)
        if val == 1:
            print(f"    {RED}[+] {feat:<25s}  conf: {conf:.2f}{RESET}")
        else:
            print(f"    {DIM}[-] {feat:<25s}  (not detected){RESET}")

    if detections:
        print(f"\n  {WHITE}{BOLD}Raw YOLO Detections:{RESET}")
        for i, det in enumerate(detections, 1):
            print(f"    {CYAN}{i}. {det['class']:<25s} conf: {det['confidence']:.4f}{RESET}")

    print(f"\n{BOLD}{'=' * 56}{RESET}\n")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cardionyx - Interactive CAD Prediction"
    )
    parser.add_argument(
        "--ecg", type=str, default=None,
        help="Path to an ECG image for YOLO analysis"
    )
    parser.add_argument(
        "--yolo-model", type=str, default=DEFAULT_YOLO_MODEL,
        help=f"Path to YOLO weights (default: {DEFAULT_YOLO_MODEL})"
    )
    parser.add_argument(
        "--xgb-model", type=str, default=DEFAULT_XGB_MODEL,
        help=f"Path to XGBoost model bundle (default: {DEFAULT_XGB_MODEL})"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.25,
        help="YOLO confidence threshold for binary features (default: 0.25)"
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'=' * 56}")
    print(f"        CARDIONYX - CAD RISK PREDICTION SYSTEM")
    print(f"{'=' * 56}{RESET}")

    # ---- Step 1: Load XGBoost model ----
    try:
        config = CADPipelineConfig(
            model_path=args.xgb_model,
            confidence_threshold=args.confidence_threshold,
        )
        predictor = CADPredictor(args.xgb_model, config=config)
        print(f"  {GREEN}XGBoost model loaded{RESET}")
    except Exception as e:
        print(f"  {RED}Failed to load XGBoost model: {e}{RESET}")
        sys.exit(1)

    # ---- Step 2: ECG image (optional) ----
    ecg_path = args.ecg
    if ecg_path is None:
        print(f"\n  {WHITE}Do you have an ECG image to analyse?{RESET}")
        response = input(f"  {CYAN}Enter image path (or press Enter to skip): {RESET}").strip()
        if response:
            ecg_path = response

    yolo_detections = []
    if ecg_path:
        yolo_detections = run_yolo_on_image(ecg_path, args.yolo_model)
        if yolo_detections:
            print(f"  {GREEN}Found {len(yolo_detections)} ECG abnormalit{'y' if len(yolo_detections) == 1 else 'ies'}{RESET}")
        else:
            print(f"  {YELLOW}No ECG abnormalities detected (or YOLO failed){RESET}")
    else:
        print(f"  {DIM}Skipping ECG image analysis{RESET}")

        # Let user manually enter ECG features
        print(f"\n{BOLD}{WHITE}--- ECG FEATURES (manual entry) ---{RESET}")
        ecg_manual_map = {
            "Q Wave":              "Pathological Q Wave (0/1)",
            "St Elevation":        "ST Elevation (0/1)",
            "St Depression":       "ST Depression (0/1)",
            "Tinversion":          "T Inversion (0/1)",
            "LVH":                 "LVH (0/1)",
            "Poor R Progression":  "Poor R Progression (Y/N)",
        }
        for col, label in ecg_manual_map.items():
            if col == "Poor R Progression":
                val = prompt_value(label, "yn", "N")
            else:
                val = prompt_value(label, "binary", 0)
            # Build a synthetic YOLO detection so the pipeline picks it up
            if (isinstance(val, int) and val == 1) or (isinstance(val, str) and val.upper() == "Y"):
                yolo_detections.append({
                    "class": col,
                    "confidence": 1.0,  # manual = certain
                })

    # ---- Step 3: Collect patient data ----
    print(f"\n{BOLD}{CYAN}--- PATIENT CLINICAL DATA ---{RESET}")
    print(f"  {DIM}Enter values at each prompt. Press Enter for defaults shown in [brackets].{RESET}")
    patient_data = collect_patient_data()

    # ---- Step 4: Run prediction ----
    print(f"\n{CYAN}Running CAD prediction...{RESET}")
    try:
        result = predictor.predict(
            yolo_detections, patient_data, return_details=True
        )
        display_results(result, yolo_detections)
    except Exception as e:
        print(f"\n{RED}Prediction failed: {e}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
