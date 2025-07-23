import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# --- Configurations ---

LATEST_ROUND_ORDER = [
    "Unattributed",
    "Pre-Seed",
    "Angel",
    "Seed",
    "Bridge",
    "Series A",
    "Series B",
    "Series C",
    "Series D",
    "Series E",
    "Convertible Note",
    "PE",
    "Conventional Debt",
]

POST_EXIT_FIELDS = [
    "Revenue_FY(23-24)_Cr",
    "EBITDA_FY(23-24)_Cr",
    "Employees_2024",
    "Revenue_FY(22-23)_Cr",
    "EBITDA_FY(22-23)_Cr",
    "Employees_2023",
    "Revenue_FY(21-22)_Cr",
    "EBITDA_FY(21-22)_Cr",
    "Employees_2022",
    "Revenue_3Y_CAGR",
]

STRUCTURAL_ZERO_FIELDS = ["Founder_Churn", "Board_Advisors"]

TARGET_COLUMNS = ["Valuation_Crores", "Years_To_Exit"]

# --- Preprocessing Pipeline ---


def build_preprocessing_pipeline(X: pd.DataFrame):
    df = X.copy()

    # Drop non-feature fields
    df = df.drop(
        columns=[col for col in ["Company_Name"] if col in df.columns], errors="ignore"
    )

    # Mask regression targets for DEADPOOLED
    if "Label" in df.columns:
        deadpooled_mask = df["Label"] == "DEADPOOLED"
        for col in TARGET_COLUMNS:
            if col in df.columns:
                df.loc[deadpooled_mask, col] = np.nan

    # Ordinal encode Latest_Funding_Round
    if "Latest_Funding_Round" in df.columns:
        ordinal_encoder = OrdinalEncoder(
            categories=[LATEST_ROUND_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df["Latest_Funding_Round_Ordinal"] = ordinal_encoder.fit_transform(
            df[["Latest_Funding_Round"]]
        )
        df.drop(columns=["Latest_Funding_Round"], inplace=True)

    # Add binary presence flags for post-exit features
    for col in POST_EXIT_FIELDS:
        if col in df.columns:
            df[f"Has_{col}"] = df[col].notna().astype(int)

    all_numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    all_categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    labels = ["Label"] + TARGET_COLUMNS

    # Split numerical columns by imputation strategy
    num_zero_cols = [col for col in STRUCTURAL_ZERO_FIELDS if col in df.columns]
    num_post_exit_cols = [col for col in POST_EXIT_FIELDS if col in df.columns]
    num_regular_cols = [
        col
        for col in all_numeric
        if col not in labels + num_zero_cols + num_post_exit_cols
    ]

    # --- Pipelines ---
    pipeline_zero = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_mean = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    pipeline_cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine all transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ("zero_fields", pipeline_zero, num_zero_cols + num_post_exit_cols),
            ("mean_fields", pipeline_mean, num_regular_cols),
            ("cat_fields", pipeline_cat, all_categorical),
        ]
    )

    final_features = (
        num_zero_cols + num_post_exit_cols + num_regular_cols + all_categorical
    )

    return preprocessor, df, final_features
