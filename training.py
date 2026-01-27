# =========================
# XGBOOST TRAINING SCRIPT
# Dataset : CAD.csv
# Target  : Cath
# =========================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 1. LOAD DATA
# -------------------------
df = pd.read_csv("CAD.csv")
print("Loaded dataset shape:", df.shape)

target_col = df.columns[-1]
print("Target column:", target_col)

# -------------------------
# 2. TARGET ENCODING ONLY
# -------------------------
if df[target_col].dtype == object:
    df[target_col] = pd.factorize(df[target_col])[0]

# -------------------------
# 3. FEATURE / TARGET SPLIT
# -------------------------
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# -------------------------
# 4. COLUMN TYPE FIX (CRITICAL)
# -------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.columns.difference(num_cols).tolist()

# Force consistency
X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
X[cat_cols] = X[cat_cols].astype(str)

print("Numeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

# -------------------------
# 5. PREPROCESSING
# -------------------------
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

# -------------------------
# 6. TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# -------------------------
# 7. XGBOOST MODEL
# -------------------------
model = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# -------------------------
# 8. TRAIN
# -------------------------
model.fit(
    X_train_p,
    y_train,
    verbose=True
)


# -------------------------
# 9. EVALUATION
# -------------------------
y_pred = model.predict(X_test_p)
y_prob = model.predict_proba(X_test_p)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# -------------------------
# 10. FEATURE IMPORTANCE
# -------------------------
feature_names = num_cols.copy()

if cat_cols:
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(cat_cols))

importances = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 20 Features:")
print(importances.head(20))

plt.figure(figsize=(10, 6))
importances.head(20).plot(kind="bar")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# -------------------------
# 11. SAVE MODEL
# -------------------------
joblib.dump(
    {"preprocessor": preprocessor, "model": model},
    "xgb_cad_model.joblib"
)

print("\nModel saved as xgb_cad_model.joblib")