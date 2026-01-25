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
