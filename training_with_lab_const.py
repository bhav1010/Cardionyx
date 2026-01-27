# ==================================================
# CORONARY ARTERY DISEASE (CAD) PREDICTION
# XGBOOST WITH MEDICAL LAB CONSTRAINTS
# AUTO-VERSIONED MODEL SAVING
# ==================================================

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from xgboost import XGBClassifier
import joblib

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
df = pd.read_csv("CAD.csv")
print("Dataset loaded:", df.shape)

# Target column
target_col = "Cath"
if df[target_col].dtype == object:
    df[target_col] = pd.factorize(df[target_col])[0]

# --------------------------------------------------
# 2. MEDICAL LAB CONSTRAINT FEATURE ENGINEERING
# (Using exact column names from CAD.csv)
# --------------------------------------------------

# Helper function
def col_exists(col):
    return col in df.columns

# FBS (>=126 mg/dL)
if col_exists("FBS"):
    df["FBS_high"] = (df["FBS"] >= 126).astype(int)

# Creatinine (0.5–1.4 mg/dL)
if col_exists("CR"):
    df["CR_abnormal"] = ((df["CR"] < 0.5) | (df["CR"] > 1.4)).astype(int)

# BUN (>20 mg/dL)
if col_exists("BUN"):
    df["BUN_high"] = (df["BUN"] > 20).astype(int)

# Hemoglobin (<12.5 g/dL – conservative)
if col_exists("HB"):
    df["HB_low"] = (df["HB"] < 12.5).astype(int)

# Platelet (150k–450k)
if col_exists("PLT"):
    df["PLT_abnormal"] = ((df["PLT"] < 150000) | (df["PLT"] > 450000)).astype(int)

# WBC (>13500)
if col_exists("WBC"):
    df["WBC_high"] = (df["WBC"] > 13500).astype(int)

# Lymphocyte % (20–40)
if col_exists("Lymph"):
    df["Lymph_abnormal"] = ((df["Lymph"] < 20) | (df["Lymph"] > 40)).astype(int)

# Neutrophil % (40–60)  ✅ actual column name = "Neut"
if col_exists("Neut"):
    df["Neutrophil_abnormal"] = ((df["Neut"] < 40) | (df["Neut"] > 60)).astype(int)

# ESR (>15)
if col_exists("ESR"):
    df["ESR_high"] = (df["ESR"] > 15).astype(int)

# Potassium (3.5–5.5)
if col_exists("K"):
    df["K_abnormal"] = ((df["K"] < 3.5) | (df["K"] > 5.5)).astype(int)

# Sodium (135–145)
if col_exists("Na"):
    df["Na_abnormal"] = ((df["Na"] < 135) | (df["Na"] > 145)).astype(int)

# LDL (>130)
if col_exists("LDL"):
    df["LDL_high"] = (df["LDL"] > 130).astype(int)

# HDL (<40)
if col_exists("HDL"):
    df["HDL_low"] = (df["HDL"] < 40).astype(int)

# Triglycerides (>150)
if col_exists("TG"):
    df["TG_high"] = (df["TG"] > 150).astype(int)

print("Medical lab constraint features added")

# --------------------------------------------------
# 3. FEATURE / TARGET SPLIT
# --------------------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# --------------------------------------------------
# 4. COLUMN TYPE HANDLING
# --------------------------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Enforce consistency
X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
X[cat_cols] = X[cat_cols].astype(str)

print("Numeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

# --------------------------------------------------
# 5. PREPROCESSING PIPELINE
# --------------------------------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# --------------------------------------------------
# 6. TRAIN–TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# --------------------------------------------------
# 7. XGBOOST MODEL
# --------------------------------------------------
model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_p, y_train)

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------
y_pred = model.predict(X_test_p)
y_prob = model.predict_proba(X_test_p)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# --------------------------------------------------
# 9. SAVE MODEL (AUTO-VERSIONED – NO OVERWRITE)
# --------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"cad_xgb_medical_model_{timestamp}.joblib"

joblib.dump(
    {
        "preprocessor": preprocessor,
        "model": model
    },
    model_filename
)

print(f"\nModel saved successfully as: {model_filename}")
