import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.predictor import (get_best_model_path, predict_from_csv,
                            predict_from_manual)
from core.trainer import train_model
from core.utils import save_uploaded_file

st.set_page_config(page_title="StartupOracle", layout="wide")
# st.title("StartupOracle")
st.markdown(
    """
    <h1 style='
        text-align: center;
        font-family: "Trebuchet MS", sans-serif;
        color: #4B8BBE;
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    '>
         StartupOracle
    </h1>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Train Models", "Predict Outcome"])

# ------------------------- TRAIN TAB -------------------------
with tabs[0]:
    st.header("Train Models")
    st.info(
        "Upload a labeled startup dataset to train classification and regression models."
    )

    train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train_csv")
    if train_file:
        path = save_uploaded_file(train_file, "data/train.csv")
        df_train = pd.read_csv(path)
        st.dataframe(df_train.head())

        with st.expander("⚙️ Select Training Configuration"):
            task_type = st.multiselect(
                "Tasks",
                ["Classification", "Valuation Regression", "Timeline Regression"],
                default=["Classification"],
            )
            model_type = st.selectbox(
                "Model Type", ["RandomForest", "XGBoost", "LightGBM"]
            )
            use_feat_sel = st.checkbox("Enable Feature Selection (K-Best)")
            manual_feat_list = st.multiselect(
                "Select Features to Include (Optional)", df_train.columns.tolist()
            )
            st.caption(
                "If you select features here, it will override automatic K-best selection."
            )

            n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
            max_depth = st.slider("max_depth", 3, 20, 6)
            learning_rate = st.slider(
                "learning_rate (Only for XGBoost/LGBM)", 0.01, 0.5, 0.1
            )
            model_note = st.text_input(
                "📝 Add a note for this training run", max_chars=100
            )

        if st.button("Train Now"):
            with st.spinner("Training models..."):
                train_model(
                    df=df_train,
                    task_type=task_type,
                    model_type=model_type,
                    use_feature_selection=use_feat_sel,
                    selected_features=manual_feat_list if manual_feat_list else None,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    model_note=model_note,
                )
            st.success(" Training complete! Models saved to /models.")

# ------------------------- PREDICTION TAB -------------------------
with tabs[1]:
    st.header("Predict Startup Outcome")

    input_mode = st.radio("Input Mode", ["Upload CSV", "Manual Input"], horizontal=True)

    try:
        classification_model_path = get_best_model_path("classification")
        feature_dtypes = joblib.load(
            os.path.join(classification_model_path, "feature_dtypes.pkl")
        )
        feature_cols = [
            feat
            for feat in feature_dtypes
            if feat not in ["Label", "Valuation_Crores", "Years_To_Exit"]
        ]

        try:
            dropdowns = joblib.load(
                os.path.join(classification_model_path, "categorical_dropdowns.pkl")
            )
        except:
            dropdowns = {}
    except Exception as e:
        st.error(f"❌ Could not load model metadata: {e}")
        st.stop()
    st.subheader("**Input Tips:**")
    st.markdown(
        """
    • Leave fields **blank** if the value is not known.  
    • Do **not** enter `0` unless the actual value is zero.  
    • For **Funding**, **Revenue**, and **EBITDA** fields:  
  - You can enter amounts like `100 M`, `USD 100M`, or `0.5B` — these will be **auto-converted to INR Crores**.  
  - If you enter a plain number (e.g., `100`) or `100 Crore`, it will be **used as-is** without conversion.
    """
    )

    if input_mode == "Upload CSV":
        sample_df = pd.DataFrame(columns=feature_cols)
        st.download_button(
            "Download Sample CSV Template",
            sample_df.to_csv(index=False).encode(),
            "sample_input_template.csv",
        )

        pred_file = st.file_uploader("Upload Your CSV", type=["csv"], key="test_csv")
        if pred_file:
            path = save_uploaded_file(pred_file, "data/predict.csv")
            df_input = pd.read_csv(path)
            st.dataframe(df_input.head())

            if st.button("Run Prediction"):
                with st.spinner("Running model inference..."):
                    preds_df, shap_values, explainer, X_df = predict_from_csv(df_input)
                    st.success("✅ Prediction complete.")
                    st.subheader("Predictions")
                    st.dataframe(preds_df)
                    st.download_button(
                        "Download Results",
                        preds_df.to_csv(index=False).encode(),
                        "predictions.csv",
                    )

    elif input_mode == "Manual Input":
        sample_input = {}
        label_columns = ["Valuation_Crores", "Years_To_Exit", "Outcome"]
        input_features = [col for col in feature_dtypes if col not in label_columns]

        with st.form("manual_input"):
            for feat in input_features:
                dtype = feature_dtypes.get(feat, "object")

                if feat in dropdowns and isinstance(dropdowns[feat], list):
                    options = dropdowns[feat]
                    dropdown = ["-- Select --"] + options + ["Other"]
                    val = st.selectbox(
                        f"{feat}", dropdown, help=f"Select or leave blank if unknown."
                    )
                    sample_input[feat] = "" if val == "-- Select --" else val
                    if val not in options and val not in ["", "-- Select --", "Other"]:
                        st.warning(f"⚠️ '{val}' was not seen during training.")
                elif dtype == "bool":
                    sample_input[feat] = (
                        1
                        if st.radio(feat, ["Yes", "No"], horizontal=True, key=feat)
                        == "Yes"
                        else 0
                    )
                else:
                    help_text = ""
                    sample_input[feat] = st.text_input(feat, help=help_text)

            submitted = st.form_submit_button("Predict")

        if submitted:
            for k, v in sample_input.items():
                if isinstance(v, str) and v.strip() == "":
                    sample_input[k] = np.nan
                elif isinstance(v, str):
                    try:
                        sample_input[k] = float(v)
                    except:
                        pass

            df_manual = pd.DataFrame([sample_input])
            for feat in feature_dtypes:
                if feat not in df_manual.columns and feat not in label_columns:
                    df_manual[feat] = np.nan
            df_manual = df_manual[
                [
                    c
                    for c in feature_dtypes
                    if c in df_manual.columns and c not in label_columns
                ]
            ]

            st.subheader("Manual Input Summary")
            st.dataframe(df_manual)

            preds_df, shap_values, explainer, X_df = predict_from_manual(df_manual)
            st.success("✅ Prediction complete.")
            st.subheader("Predictions")
            st.dataframe(preds_df)
            st.download_button(
                "Download Results",
                preds_df.to_csv(index=False).encode(),
                "predictions.csv",
            )
