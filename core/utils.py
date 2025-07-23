import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from xgboost import XGBClassifier, XGBRegressor

# Trainer helper functions for model training, feature selection, and versioning


def get_model(model_type, task, n_estimators, max_depth, learning_rate):
    if task == "classification":
        if model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        elif model_type == "XGBoost":
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )
        elif model_type == "LightGBM":
            return LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )
    else:
        if model_type == "RandomForest":
            return RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        elif model_type == "XGBoost":
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )
        elif model_type == "LightGBM":
            return LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )


def apply_feature_selection(X, y, task, k=18):
    score_func = f_classif if task == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    return X_new, selector


def create_version_folder(base_dir="models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(version_path, exist_ok=True)
    return version_path


def update_best_model(model_path, task, score):
    task_map = {
        "classification": "best_model_classification.txt",
        "valuation": "best_model_valuation.txt",
        "timeline": "best_model_timeline.txt",
    }
    best_path_file = os.path.join("models", task_map[task])
    joblib.dump(score, os.path.join(model_path, f"{task}_score.pkl"))
    with open(best_path_file, "w") as f:
        f.write(os.path.basename(model_path))


def get_current_best_score(task):
    task_map = {
        "classification": "best_model_classification.txt",
        "valuation": "best_model_valuation.txt",
        "timeline": "best_model_timeline.txt",
    }
    try:
        with open(os.path.join("models", task_map[task]), "r") as f:
            best_model_dir = f.read().strip()
        score_file = os.path.join("models", best_model_dir, f"{task}_score.pkl")
        return joblib.load(score_file)
    except:
        return float("-inf") if task != "classification" else 0


def extract_and_save_categorical_dropdowns(preprocessor, version_path):
    try:
        for name, transformer, cols in preprocessor.transformers_:
            if (
                hasattr(transformer, "named_steps")
                and "encoder" in transformer.named_steps
            ):
                encoder = transformer.named_steps["encoder"]
                cat_dict = {
                    cols[i]: list(encoder.categories_[i]) for i in range(len(cols))
                }
                joblib.dump(
                    cat_dict, os.path.join(version_path, "categorical_dropdowns.pkl")
                )
                break
        else:
            print("⚠️ Could not extract dropdown categories: encoder not found.")
    except Exception as e:
        print(f"⚠️ Failed to extract dropdown categories: {e}")


# File handling utilities


def save_uploaded_file(uploadedfile, save_path):
    with open(save_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return save_path


def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def get_latest_model_path(base_dir="models"):
    if not os.path.exists(base_dir):
        return None
    runs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("run_")], reverse=True
    )
    return os.path.join(base_dir, runs[0]) if runs else None


def get_latest_model_file(filename, selected_model=None):
    base_dir = (
        os.path.join("models", selected_model)
        if selected_model
        else get_latest_model_path()
    )
    if base_dir is None:
        return None
    path = os.path.join(base_dir, filename)
    return path if os.path.exists(path) else None


# Funding Conversion


def convert_million_dollars_to_cr(value):
    """
    Convert values like '100 M', 'USD 200M', '0.5B' to INR Crores.
    If already float (numeric), returns as-is.
    """
    try:
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            value = (
                value.replace("$", "")
                .replace("USD", "")
                .replace(",", "")
                .strip()
                .upper()
            )
            if "M" in value:
                num = float(value.replace("M", ""))
                return round(num * 83 / 10, 2)
            elif "B" in value:
                num = float(value.replace("B", ""))
                return round(num * 83 * 100 / 10, 2)
            elif "CRORE" in value:
                return float(value.replace("CRORE", "").strip())
        return float(value)
    except:
        return np.nan


# Column Type Casting


def cast_column_types(df, feature_dtypes):
    for col in df.columns:
        if col in feature_dtypes:
            try:
                if feature_dtypes[col] in ["int64", "float64"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(str)
            except:
                pass
    return df


# Funding Conversion Application


def apply_funding_conversions(df):
    for col in df.columns:
        if df[col].dtype == "object":
            if any(
                keyword in col.upper()
                for keyword in ["FUNDING", "REVENUE", "EBITDA", "AMOUNT"]
            ):
                df[col] = df[col].apply(convert_million_dollars_to_cr)
    return df
