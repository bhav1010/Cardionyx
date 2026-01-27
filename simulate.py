import joblib
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ecg_stream import get_ecg_features

# =========================
# ANSI COLORS
# =========================
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
BOLD = "\033[1m"

# =========================
# LOAD MODEL (SECONDARY)
# =========================
bundle = joblib.load("xgb_cad_model.joblib")
preprocessor = bundle["preprocessor"]
model = bundle["model"]

print(f"\n{GREEN}✔ Model loaded (secondary risk estimator){RESET}")
print(f"{CYAN}🫀 Starting real-time heart monitoring...{RESET}\n")

# =========================
# LOAD SCHEMA
# =========================
schema = pd.read_csv("CAD.csv")
feature_columns = schema.drop(columns=[schema.columns[-1]]).columns

# =========================
# ALERT HANDLERS
# =========================
def notify_family(msg):
    print(f"{YELLOW}{BOLD}📢 FAMILY ALERT → {msg}{RESET}")

def notify_hospital(msg):
    print(f"{RED}{BOLD}🚑 HOSPITAL ALERT → {msg}{RESET}")

# =========================
# PHYSIOLOGICAL RISK ENGINE
# (FIXED TO SHOW WARNING)
# =========================
def physiological_risk(features):
    score = 0
    reasons = []

    # ---- Mild risk (WARNING entry zone)
    if features["RestingBP"] > 145:
        score += 1
        reasons.append("Elevated blood pressure")

    if features["ExerciseAngina"] == 1:
        score += 1
        reasons.append("Exercise-induced angina")

    # ---- Moderate risk
    if features["Oldpeak"] > 1.5:
        score += 2
        reasons.append("ST depression")

    # ---- Severe risk (SOS)
    if features["RestingBP"] > 170:
        score += 2
        reasons.append("Severe hypertension")

    if features["MaxHR"] < 50 or features["MaxHR"] > 160:
        score += 3
        reasons.append("Dangerous heart rate")

    return score, reasons

# =========================
# STATE MEMORY
# =========================
warning_counter = 0
sos_counter = 0
WARNING_THRESHOLD = 2
SOS_THRESHOLD = 2

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        # -------------------------
        # GET SIMULATED ECG FEATURES
        # -------------------------
        features = get_ecg_features()

        # -------------------------
        # BUILD ML INPUT
        # -------------------------
        patient_full = schema.drop(columns=[schema.columns[-1]]).iloc[[0]].astype(object)
        patient_full.iloc[0] = np.nan

        for k, v in features.items():
            if k in patient_full.columns:
                patient_full.loc[0, k] = v

        # -------------------------
        # ML INFERENCE (SECONDARY)
        # -------------------------
        X_processed = preprocessor.transform(patient_full)
        ml_probability = model.predict_proba(X_processed)[0][1]

        # -------------------------
        # PHYSIOLOGICAL DECISION
        # -------------------------
        phys_score, reasons = physiological_risk(features)

        if phys_score >= 5:
            state = "SOS"
            color = RED
        elif phys_score >= 3:
            state = "WARNING"
            color = YELLOW
        else:
            state = "NORMAL"
            color = GREEN

        # -------------------------
        # ESCALATION LOGIC
        # -------------------------
        if state == "WARNING":
            warning_counter += 1
            sos_counter = 0

            if warning_counter == WARNING_THRESHOLD:
                notify_family(", ".join(reasons))

        elif state == "SOS":
            sos_counter += 1
            warning_counter = 0

            if sos_counter == SOS_THRESHOLD:
                notify_family(", ".join(reasons))
                notify_hospital(", ".join(reasons))

        else:
            warning_counter = 0
            sos_counter = 0

        # -------------------------
        # DISPLAY
        # -------------------------
        print(f"\n{color}{BOLD}================ HEART MONITOR ================={RESET}")
        print(f"{WHITE}Time                  : {datetime.now().isoformat()}{RESET}")
        print(f"{color}{BOLD}System State           : {state}{RESET}")
        print(f"{CYAN}Physiological Risk     : {phys_score}{RESET}")
        print(f"{CYAN}ML Long-term Risk      : {ml_probability*100:.2f}%{RESET}")

        print(f"{WHITE}Detected Features:{RESET}")
        for k, v in features.items():
            print(f"  {CYAN}• {k:<15}:{RESET} {v}")

        if reasons:
            print(f"{color}{BOLD}⚠ Risk Reasons:{RESET}")
            for r in reasons:
                print(f"   {color}- {r}{RESET}")

        print(f"{color}{BOLD}================================================{RESET}")

        time.sleep(2)

except KeyboardInterrupt:
    print(f"\n{WHITE}🛑 Monitoring stopped by user.{RESET}")
    print(f"{GREEN}✅ System shut down safely.{RESET}")
