import os
from math import sqrt

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core.preprocessing import build_preprocessing_pipeline
from core.utils import (apply_feature_selection, create_version_folder,
                        extract_and_save_categorical_dropdowns,
                        get_current_best_score, get_model, update_best_model)

# ------------------ Training Functions ------------------


def train_classification_model(
    df,
    model_type,
    n_estimators,
    max_depth,
    learning_rate,
    do_feature_selection,
    version_path,
):
    df = df[df["Label"].isin(["IPO", "M&A", "DEADPOOLED"])]
    X = df.drop(columns=["Label", "Valuation_Crores", "Years_To_Exit"], errors="ignore")
    y = df["Label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, os.path.join(version_path, "label_encoder.pkl"))

    X_raw = df.drop(
        columns=["Label", "Valuation_Crores", "Years_To_Exit"], errors="ignore"
    )
    joblib.dump(
        X_raw.columns.tolist(), os.path.join(version_path, "raw_input_columns.pkl")
    )
    preprocessor, X_processed_df, _ = build_preprocessing_pipeline(X_raw)
    preprocessor.fit(X_processed_df)
    extract_and_save_categorical_dropdowns(preprocessor, version_path)
    X_processed = preprocessor.transform(X_processed_df)

    feature_names = preprocessor.get_feature_names_out()
    joblib.dump(
        feature_names.tolist(),
        os.path.join(version_path, "classification_features.pkl"),
    )
    joblib.dump(preprocessor, os.path.join(version_path, "preprocessor.pkl"))

    if do_feature_selection:
        X_processed, selector = apply_feature_selection(
            X_processed, y_encoded, task="classification"
        )
        feature_names = preprocessor.get_feature_names_out()
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        joblib.dump(
            selected_feature_names,
            os.path.join(version_path, "classification_features.pkl"),
        )

    model = get_model(
        model_type, "classification", n_estimators, max_depth, learning_rate
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_dict = {"labels": le.classes_.tolist(), "matrix": cm.tolist()}

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    fig_path = os.path.join(version_path, "classification_confusion_matrix.png")
    plt.savefig(fig_path)
    plt.close()

    accuracy = accuracy_score(y_test, y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    joblib.dump(
        {"accuracy": accuracy, "classification_report": report},
        os.path.join(version_path, "classification_scores.pkl"),
    )

    joblib.dump(model, os.path.join(version_path, "classification_model.pkl"))

    # Check and update best model
    if accuracy > get_current_best_score("classification"):
        update_best_model(version_path, "classification", accuracy)
        print("This is now the BEST model.")


def train_regression_model(
    df,
    target_column,
    model_type,
    n_estimators,
    max_depth,
    learning_rate,
    do_feature_selection,
    version_path,
):
    df = df[df["Label"].isin(["IPO", "M&A"])]
    df = df.dropna(subset=[target_column])
    X = df.drop(columns=["Label", "Valuation_Crores", "Years_To_Exit"])
    y = df[target_column]

    X_raw = df.drop(columns=["Label", "Valuation_Crores", "Years_To_Exit"])
    joblib.dump(
        X_raw.columns.tolist(), os.path.join(version_path, "raw_input_columns.pkl")
    )
    preprocessor, X_processed_df, _ = build_preprocessing_pipeline(X_raw)
    preprocessor.fit(X_processed_df)
    extract_and_save_categorical_dropdowns(preprocessor, version_path)
    X_processed = preprocessor.transform(X_processed_df)

    suffix = "valuation" if target_column == "Valuation_Crores" else "timeline"
    joblib.dump(preprocessor, os.path.join(version_path, f"preprocessor.pkl"))

    if do_feature_selection:
        X_processed, selector = apply_feature_selection(
            X_processed, y, task="regression"
        )
        feature_names = preprocessor.get_feature_names_out()
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        joblib.dump(
            selected_feature_names, os.path.join(version_path, f"{suffix}_features.pkl")
        )

    model = get_model(model_type, "regression", n_estimators, max_depth, learning_rate)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nRegression Report for {target_column}:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    joblib.dump(metrics, os.path.join(version_path, f"{suffix}_scores.pkl"))

    task = "valuation" if target_column == "Valuation_Crores" else "timeline"
    if metrics["r2"] > get_current_best_score(task):
        update_best_model(version_path, task, metrics["r2"])

    joblib.dump(model, os.path.join(version_path, f"{suffix}_model.pkl"))
    feature_names = preprocessor.get_feature_names_out()
    joblib.dump(
        feature_names.tolist(), os.path.join(version_path, f"{suffix}_features.pkl")
    )


# ------------------ Unified Training Entry ------------------


def train_model(
    df,
    task_type,
    model_type,
    use_feature_selection=False,
    selected_features=None,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    model_note: str = "",
):
    df = df.copy()
    if selected_features:
        selected_features = list(set(selected_features) & set(df.columns))
        df = df[selected_features + ["Label", "Valuation_Crores", "Years_To_Exit"]]

    version_path = create_version_folder()
    joblib.dump(selected_features, os.path.join(version_path, "selected_features.pkl"))
    if model_note:
        with open(os.path.join(version_path, "note.txt"), "w") as f:
            f.write(model_note.strip())

    if "Classification" in task_type:
        train_classification_model(
            df,
            model_type,
            n_estimators,
            max_depth,
            learning_rate,
            do_feature_selection=use_feature_selection,
            version_path=version_path,
        )

    if "Valuation Regression" in task_type:
        train_regression_model(
            df,
            "Valuation_Crores",
            model_type,
            n_estimators,
            max_depth,
            learning_rate,
            do_feature_selection=use_feature_selection,
            version_path=version_path,
        )

    if "Timeline Regression" in task_type:
        train_regression_model(
            df,
            "Years_To_Exit",
            model_type,
            n_estimators,
            max_depth,
            learning_rate,
            do_feature_selection=use_feature_selection,
            version_path=version_path,
        )

    print(f"\n All models saved to: {version_path}")
    try:
        feature_columns = df.drop(
            columns=["Label", "Valuation_Crores", "Years_To_Exit"], errors="ignore"
        )
        feature_dtypes = feature_columns.dtypes.astype(str).to_dict()
        joblib.dump(feature_dtypes, os.path.join(version_path, "feature_dtypes.pkl"))
    except Exception as e:
        print(f"⚠️ Could not save feature dtypes: {e}")
