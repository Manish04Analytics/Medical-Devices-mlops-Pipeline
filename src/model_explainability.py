"""
Healthcare Medical-Device Sales
Step 8: Model Explainability & Feature Importance
===================================================

Purpose
-------
Analyze which business and historical-demand features
drive the Gradient Boosting model's predictions.

Outputs
-------
outputs/
    feature_importance.csv
    feature_importance_top20.png
    explainability_report.txt
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingRegressor


warnings.filterwarnings("ignore")


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

TRAIN_FILE = (
    DATA_DIR
    / "train.csv"
)

VALIDATION_FILE = (
    DATA_DIR
    / "validation.csv"
)

TEST_FILE = (
    DATA_DIR
    / "test.csv"
)

IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "feature_importance.csv"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "feature_importance_top20.png"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "explainability_report.txt"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Units_Sold"


# ============================================================
# EXCLUDED FEATURES
# ============================================================

EXCLUDED_COLUMNS = [

    "Units_Sold",

    "Transaction_Date",

    "Hospital_ID",
    "Hospital_Name",

    "Product_ID",
    "Product_Name",

    "Total_Revenue",
    "Calculated_Revenue",
    "Revenue_Difference",

    "Daily_Units_Sold",
    "Daily_Revenue",

    "Hospital_Total_Demand",
    "Hospital_Demand_Share",
    "Hospital_Product_Diversity",

    "Product_Total_Demand",
    "Product_Demand_Share",
    "Product_Hospital_Coverage",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\n📂 Loading model-ready datasets..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"✅ Train rows      : {len(train_df):,}"
    )

    print(
        f"✅ Validation rows : {len(validation_df):,}"
    )

    print(
        f"✅ Test rows       : {len(test_df):,}"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train_df,
    validation_df,
    test_df,
):

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    feature_columns = [
        column
        for column in feature_columns
        if column in validation_df.columns
        and column in test_df.columns
    ]

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    X_validation = validation_df[
        feature_columns
    ].copy()

    y_validation = validation_df[
        TARGET
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    )


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(
    X_train,
):

    numerical_features = (
        X_train
        .select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32",
            ]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    return (
        numerical_features,
        categorical_features,
    )


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(
    numerical_features,
    categorical_features,
):

    numerical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],

        remainder="drop",
    )

    return preprocessor


# ============================================================
# TRAIN BEST MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    preprocessor,
):

    print(
        "\n🤖 Training Gradient Boosting model..."
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor,
            ),

            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "✅ Gradient Boosting model trained."
    )

    return pipeline


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_feature_names(
    pipeline,
):

    preprocessor = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        feature_names = [
            name.replace(
                "numerical__",
                ""
            )
            .replace(
                "categorical__",
                ""
            )
            for name in feature_names
        ]

    except Exception:

        feature_names = [
            f"Feature_{i}"
            for i in range(
                len(
                    pipeline
                    .named_steps[
                        "model"
                    ]
                    .feature_importances_
                )
            )
        ]

    return feature_names


# ============================================================
# CALCULATE IMPORTANCE
# ============================================================

def calculate_importance(
    pipeline,
):

    model = (
        pipeline
        .named_steps[
            "model"
        ]
    )

    feature_names = (
        get_feature_names(
            pipeline
        )
    )

    importances = (
        model
        .feature_importances_
    )

    if len(feature_names) != len(
        importances
    ):

        raise ValueError(
            "❌ Feature-name count does not "
            "match importance count."
        )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance_df[
        "Importance_Percent"
    ] = (
        importance_df[
            "Importance"
        ]
        * 100
    )

    importance_df[
        "Cumulative_Importance"
    ] = (
        importance_df[
            "Importance"
        ]
        .cumsum()
    )

    return importance_df


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

def display_top_features(
    importance_df,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "🔥 TOP 20 DEMAND-DRIVING FEATURES"
    )

    print(
        "=" * 70
    )

    top20 = (
        importance_df
        .head(20)
    )

    for index, row in (
        top20
        .iterrows()
    ):

        print(
            f"{index + 1:>2}. "
            f"{row['Feature']:<40} "
            f"{row['Importance_Percent']:.2f}%"
        )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    importance_df,
):

    print(
        "\n📊 Creating feature-importance visualization..."
    )

    top20 = (
        importance_df
        .head(20)
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(12, 9)
    )

    plt.barh(
        top20["Feature"],
        top20["Importance"],
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Features Driving Medical-Device Demand"
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"✅ Visualization saved:"
        f"\n   {PLOT_FILE}"
    )


# ============================================================
# SAVE IMPORTANCE
# ============================================================

def save_importance(
    importance_df,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"\n💾 Feature importance saved:"
        f"\n   {IMPORTANCE_FILE}"
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    importance_df,
    pipeline,
    X_test,
    y_test,
):

    predictions = pipeline.predict(
        X_test
    )

    residuals = (
        y_test
        - predictions
    )

    mae = np.mean(
        np.abs(
            residuals
        )
    )

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    report = []

    report.append(
        "HEALTHCARE MEDICAL-DEVICE SALES"
    )

    report.append(
        "MODEL EXPLAINABILITY REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append(
        ""
    )

    report.append(
        "MODEL"
    )

    report.append(
        "Gradient Boosting Regressor"
    )

    report.append(
        ""
    )

    report.append(
        "TEST SET DIAGNOSTICS"
    )

    report.append(
        f"MAE  : {mae:.4f}"
    )

    report.append(
        f"RMSE : {rmse:.4f}"
    )

    report.append(
        ""
    )

    report.append(
        "TOP 20 FEATURES"
    )

    report.append(
        "-" * 70
    )

    for index, row in (
        importance_df
        .head(20)
        .iterrows()
    ):

        report.append(
            f"{index + 1}. "
            f"{row['Feature']} "
            f"({row['Importance_Percent']:.2f}%)"
        )

    report.append(
        ""
    )

    report.append(
        "INTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "Feature importance indicates how strongly "
        "each feature contributes to the Gradient "
        "Boosting model's predictive decisions."
    )

    report.append(
        ""
    )

    report.append(
        "IMPORTANT:"
    )

    report.append(
        "Feature importance does not imply causality."
    )

    report.append(
        "Business interpretation should therefore "
        "be combined with domain knowledge and "
        "additional explainability techniques."
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                report
            )
        )

    print(
        f"\n📝 Explainability report saved:"
        f"\n   {REPORT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "🧠 HEALTHCARE MEDICAL-DEVICE SALES"
    )

    print(
        "MODEL EXPLAINABILITY PIPELINE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_features(
        train_df,
        validation_df,
        test_df,
    )

    print(
        f"\n🔢 Features used: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    (
        numerical_features,
        categorical_features,
    ) = identify_feature_types(
        X_train
    )

    print(
        f"Numerical features   : "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_features)}"
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = (
        build_preprocessor(
            numerical_features,
            categorical_features,
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline = train_model(
        X_train,
        y_train,
        preprocessor,
    )

    # --------------------------------------------------------
    # Importance
    # --------------------------------------------------------

    importance_df = (
        calculate_importance(
            pipeline
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_top_features(
        importance_df
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_importance(
        importance_df
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    create_visualization(
        importance_df
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    generate_report(
        importance_df,
        pipeline,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ STEP 8 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\n📁 Generated files:"
    )

    print(
        f"   • {IMPORTANCE_FILE}"
    )

    print(
        f"   • {PLOT_FILE}"
    )

    print(
        f"   • {REPORT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

        sys.exit(0)

    except Exception as exc:

        print(
            "\n❌ MODEL EXPLAINABILITY FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)