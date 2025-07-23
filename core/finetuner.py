import os
from datetime import datetime
from math import sqrt

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from preprocessing import build_preprocessing_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# ------------------------ CONFIG ------------------------

MODELS = {
    "classification": {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(),
    },
    "regression": {
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
    },
}

PARAM_GRID = {
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.1, 0.3],
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.1, 0.3],
    },
}

# ------------------------ UTILS ------------------------


def create_version_folder(base_dir="models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(version_path, exist_ok=True)
    return version_path


def update_best_model_file(task, folder, score):
    with open(f"models/best_model_{task}.txt", "w") as f:
        f.write(os.path.basename(folder))
    joblib.dump(score, os.path.join(folder, f"{task}_score.pkl"))


# ------------------------ MAIN TRAINING ------------------------


def finetune_task(df, task):
    if task == "classification":
        df = df[df["Label"].isin(["IPO", "M&A", "DEADPOOLED"])]
        X = df.drop(
            columns=["Label", "Valuation_Crores", "Years_To_Exit"], errors="ignore"
        )
        y = df["Label"]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif task == "valuation":
        df = df[df["Label"].isin(["IPO", "M&A"])]
        df = df.dropna(subset=["Valuation_Crores"])
        X = df.drop(columns=["Label", "Valuation_Crores", "Years_To_Exit"])
        y = df["Valuation_Crores"]
    else:
        df = df[df["Label"].isin(["IPO", "M&A"])]
        df = df.dropna(subset=["Years_To_Exit"])
        X = df.drop(columns=["Label", "Valuation_Crores", "Years_To_Exit"])
        y = df["Years_To_Exit"]

    preprocessor, X_processed_df, _ = build_preprocessing_pipeline(X)
    X_processed = preprocessor.fit_transform(X_processed_df)

    # Train/Val/Test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_processed, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42
    )

    best_model = None
    best_score = -np.inf
    best_model_name = ""
    best_params = {}
    best_folder = ""

    task_type = "classification" if task == "classification" else "regression"

    for model_name, model in MODELS[task_type].items():
        print(f"\n🔍 Tuning {model_name} for {task}...")

        grid = GridSearchCV(
            model,
            PARAM_GRID[model_name],
            cv=3,
            scoring="accuracy" if task_type == "classification" else "r2",
        )
        grid.fit(X_train, y_train)

        print(f"Best Params for {model_name}: {grid.best_params_}")
        print(f"Validation Score: {grid.best_score_:.4f}")

        # Retrain on Train+Val
        best_model_current = grid.best_estimator_
        best_model_current.fit(X_trainval, y_trainval)

        # Evaluate on Test
        y_pred = best_model_current.predict(X_test)

        score = (
            accuracy_score(y_test, y_pred)
            if task_type == "classification"
            else r2_score(y_test, y_pred)
        )
        print(f"🧪 Test Score: {score:.4f}")

        if score > best_score:
            best_model = best_model_current
            best_score = score
            best_model_name = model_name
            best_params = grid.best_params_

            folder = create_version_folder()
            best_folder = folder

            # Save common artifacts
            joblib.dump(best_model, os.path.join(folder, f"{task}_model.pkl"))
            joblib.dump(preprocessor, os.path.join(folder, f"{task}_preprocessor.pkl"))
            feature_names = preprocessor.get_feature_names_out()
            joblib.dump(
                feature_names.tolist(), os.path.join(folder, f"{task}_features.pkl")
            )

            if task == "classification":
                joblib.dump(
                    label_encoder, os.path.join(folder, f"{task}_label_encoder.pkl")
                )
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                joblib.dump(cm, os.path.join(folder, f"{task}_confusion_matrix.pkl"))
                joblib.dump(
                    report, os.path.join(folder, f"{task}_classification_scores.pkl")
                )
            else:
                metrics = {
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": sqrt(mean_squared_error(y_test, y_pred)),
                    "r2": r2_score(y_test, y_pred),
                }
                joblib.dump(metrics, os.path.join(folder, f"{task}_scores.pkl"))

            # Save dtype map for manual UI prediction
            feature_dtypes = X.dtypes.astype(str).to_dict()
            joblib.dump(
                feature_dtypes, os.path.join(folder, f"{task}_feature_dtypes.pkl")
            )

            with open(os.path.join(folder, "note.txt"), "w") as f:
                f.write(
                    f"Model: {model_name}\nParams: {best_params}\nScore: {score:.4f}\n"
                )

    update_best_model_file(task, best_folder, best_score)
    print(f"\nBest model for {task}: {best_model_name} (score: {best_score:.4f})")
    print(f" Saved to: {best_folder}")


# ------------------------ ENTRY ------------------------

if __name__ == "__main__":
    df = pd.read_csv("data/startup_dataset.csv")

    print("\n--- Starting Finetuning for All Tasks ---")
    for task in ["classification", "valuation", "timeline"]:
        finetune_task(df, task)
