import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from core.preprocessing import build_preprocessing_pipeline
from core.utils import apply_funding_conversions, cast_column_types

MODEL_BASE_DIR = "models"


def get_best_model_path(task):
    file_map = {
        "classification": "best_model_classification.txt",
        "valuation": "best_model_valuation.txt",
        "timeline": "best_model_timeline.txt",
    }
    path_file = os.path.join(MODEL_BASE_DIR, file_map[task])
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            best_version = f.read().strip()
        return os.path.join(MODEL_BASE_DIR, best_version)
    else:
        raise FileNotFoundError(f"❌ No best model found for task: {task}")


# Predict from CSV


def predict_from_csv(df):
    df = df.copy()
    df = apply_funding_conversions(df)
    for col in ["Valuation_Crores", "Years_To_Exit"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- CLASSIFICATION ---
    class_model_path = get_best_model_path("classification")
    df_class = df.drop(columns=["Label"], errors="ignore")

    try:
        feature_dtypes = joblib.load(
            os.path.join(class_model_path, "feature_dtypes.pkl")
        )
        df_class = cast_column_types(df_class, feature_dtypes)
    except Exception as e:
        st.warning(f"⚠️ Could not cast input columns: {e}")

    preprocessor = joblib.load(os.path.join(class_model_path, "preprocessor.pkl"))
    _, df_processed, _ = build_preprocessing_pipeline(df_class)
    X_transformed = preprocessor.transform(df_processed)

    feature_names = preprocessor.get_feature_names_out()

    selector_path = os.path.join(
        class_model_path, "classification_feature_selector.pkl"
    )
    if os.path.exists(selector_path):
        selector = joblib.load(selector_path)
        try:
            X_transformed = selector.transform(X_transformed)
            feature_names = feature_names[selector.get_support()]
        except:
            st.warning("⚠️ Feature selector incompatible. Skipping.")

    clf_model = joblib.load(os.path.join(class_model_path, "classification_model.pkl"))
    try:
        label_encoder = joblib.load(
            os.path.join(class_model_path, "classification_label_encoder.pkl")
        )
    except FileNotFoundError:
        label_encoder = joblib.load(os.path.join(class_model_path, "label_encoder.pkl"))

    y_class_encoded = clf_model.predict(X_transformed)
    y_class_probs = clf_model.predict_proba(X_transformed)
    y_class_labels = label_encoder.inverse_transform(y_class_encoded)

    preds_df = pd.DataFrame(
        {
            "Predicted_Outcome": y_class_labels,
            "IPO_Prob": y_class_probs[:, list(label_encoder.classes_).index("IPO")],
            "M&A_Prob": y_class_probs[:, list(label_encoder.classes_).index("M&A")],
            "DEADPOOLED_Prob": y_class_probs[
                :, list(label_encoder.classes_).index("DEADPOOLED")
            ],
        }
    )

    # SHAP Explainability
    try:
        explainer = shap.TreeExplainer(clf_model)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        shap_values = None
        explainer = None
        st.warning(f"⚠️ SHAP explainability failed: {e}")

    # --- REGRESSION ---
    for task in ["valuation", "timeline"]:
        reg_model_path = get_best_model_path(task)
        model_file = os.path.join(reg_model_path, f"{task}_model.pkl")
        preproc_file = os.path.join(reg_model_path, "preprocessor.pkl")
        fs_file = os.path.join(reg_model_path, f"{task}_feature_selector.pkl")

        if not os.path.exists(model_file):
            continue

        model = joblib.load(model_file)
        pre = joblib.load(preproc_file)
        fs = joblib.load(fs_file) if os.path.exists(fs_file) else None

        indices = preds_df.index[preds_df["Predicted_Outcome"].isin(["IPO", "M&A"])]
        df_subset = df.iloc[indices]

        if df_subset.empty:
            st.info(f"ℹ️ Skipping {task} regression — no IPO/M&A rows.")
            continue

        try:
            df_subset_casted = cast_column_types(df_subset, feature_dtypes)
            _, df_reg_processed, _ = build_preprocessing_pipeline(df_subset_casted)
            X_reg = pre.transform(df_reg_processed)
            if fs:
                X_reg = fs.transform(X_reg)
        except:
            st.warning(
                f"⚠️ Preprocessing or feature selection failed for {task}. Skipping."
            )
            continue

        y_pred = model.predict(X_reg)
        y_low = (y_pred * 0.9).round(1)
        y_high = (y_pred * 1.1).round(1)

        if task == "valuation":
            preds_df.loc[indices, "Valuation_Cr"] = [
                f"₹{lo}–₹{hi} Cr" for lo, hi in zip(y_low, y_high)
            ]
        else:
            preds_df.loc[indices, "Years_To_Exit"] = [
                f"{lo}–{hi} years" for lo, hi in zip(y_low, y_high)
            ]

    if "Valuation_Cr" not in preds_df.columns:
        preds_df["Valuation_Cr"] = np.nan
    if "Years_To_Exit" not in preds_df.columns:
        preds_df["Years_To_Exit"] = np.nan

    try:
        X_df = pd.DataFrame(X_transformed, columns=feature_names, index=df_class.index)
    except:
        X_df = pd.DataFrame(X_transformed, index=df_class.index)

    return preds_df, shap_values, explainer, X_df


# Predict from manual input


def predict_from_manual(df_input):
    df = df_input.copy()
    df = apply_funding_conversions(df)

    for col in ["Valuation_Crores", "Years_To_Exit"]:
        if col not in df.columns:
            df[col] = np.nan

    try:
        class_model_path = get_best_model_path("classification")
        feature_dtypes = joblib.load(
            os.path.join(class_model_path, "feature_dtypes.pkl")
        )
        raw_input_columns = joblib.load(
            os.path.join(class_model_path, "raw_input_columns.pkl")
        )
        preprocessor = joblib.load(os.path.join(class_model_path, "preprocessor.pkl"))

        df = cast_column_types(df, feature_dtypes)

        for col in raw_input_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[raw_input_columns]

        return predict_from_csv(df)

    except Exception as e:
        st.error(f"❌ Error in manual prediction pipeline: {e}")
        return pd.DataFrame(), None, None, pd.DataFrame()
