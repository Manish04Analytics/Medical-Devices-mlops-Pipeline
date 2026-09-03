"""
Healthcare Medical-Device Sales
Step 7: Baseline Machine Learning Models
=========================================

Objective
---------
Predict Units_Sold using historical demand, inventory,
commercial, market and hospital/product characteristics.

Models
------
1. Naive historical baseline
2. Linear Regression
3. Random Forest
4. Gradient Boosting

Evaluation
----------
MAE
RMSE
R2
MAPE

Important
---------
The project uses a chronological train/validation/test split.

Potentially leakage-prone aggregate/current-period features
are deliberately excluded from the first modeling stage.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "model_comparison.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Units_Sold"


# ============================================================
# COLUMNS TO EXCLUDE
# ============================================================

# These columns are excluded from the initial model because
# they are identifiers, dates, direct target transformations,
# or potentially leakage-prone aggregate/current-period metrics.

EXCLUDED_COLUMNS = [

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    "Units_Sold",

    # --------------------------------------------------------
    # Raw date
    # --------------------------------------------------------

    "Transaction_Date",

    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------

    "Hospital_ID",
    "Hospital_Name",
    "Product_ID",
    "Product_Name",

    # --------------------------------------------------------
    # Direct target-derived variables
    # --------------------------------------------------------

    "Total_Revenue",
    "Calculated_Revenue",
    "Revenue_Difference",

    "Daily_Units_Sold",
    "Daily_Revenue",

    # --------------------------------------------------------
    # Potentially leakage-prone aggregate features
    # --------------------------------------------------------

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

def load_datasets():

    print("\n📂 Loading model-ready datasets...")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"❌ Training file not found:\n{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"❌ Validation file not found:\n{VALIDATION_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"❌ Test file not found:\n{TEST_FILE}"
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
        f"✅ Train      : {len(train_df):,} rows"
    )

    print(
        f"✅ Validation : {len(validation_df):,} rows"
    )

    print(
        f"✅ Test       : {len(test_df):,} rows"
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

    print(
        "\n🧹 Preparing model features..."
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if TARGET not in train_df.columns:
        raise ValueError(
            f"❌ Target column '{TARGET}' not found."
        )

    # --------------------------------------------------------
    # Determine usable feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    # --------------------------------------------------------
    # Make sure columns exist in all datasets
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in feature_columns
        if column in validation_df.columns
        and column in test_df.columns
    ]

    print(
        f"Total usable features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # X / y split
    # --------------------------------------------------------

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

    print(
        "\n🔢 NUMERICAL FEATURES"
    )

    print(
        f"Count: {len(numerical_features)}"
    )

    for feature in numerical_features:
        print(
            f"   • {feature}"
        )

    print(
        "\n🔤 CATEGORICAL FEATURES"
    )

    print(
        f"Count: {len(categorical_features)}"
    )

    for feature in categorical_features:
        print(
            f"   • {feature}"
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
# BUILD MODELS
# ============================================================

def build_models(
    preprocessor,
):

    models = {}

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    models[
        "Linear Regression"
    ] = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor,
            ),

            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models[
        "Random Forest"
    ] = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor,
            ),

            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    models[
        "Gradient Boosting"
    ] = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor,
            ),

            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_leaf=3,
                    random_state=42,
                ),
            ),
        ]
    )

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    y_true_array = np.asarray(
        y_true
    )

    predictions_array = np.asarray(
        predictions
    )

    non_zero_mask = (
        y_true_array != 0
    )

    if non_zero_mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    y_true_array[
                        non_zero_mask
                    ]
                    -
                    predictions_array[
                        non_zero_mask
                    ]
                )
                /
                y_true_array[
                    non_zero_mask
                ]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_Percent": mape,
    }


# ============================================================
# NAIVE BASELINE
# ============================================================

def evaluate_naive_baseline(
    y_train,
    y_validation,
):

    print(
        "\n📏 Creating naive baseline..."
    )

    # Predict the historical training median
    # for every validation observation.

    baseline_prediction = np.repeat(
        y_train.median(),
        len(y_validation),
    )

    metrics = calculate_metrics(
        y_validation,
        baseline_prediction,
    )

    print(
        "\nNAIVE BASELINE"
    )

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    print(
        f"MAPE : {metrics['MAPE_Percent']:.2f}%"
    )

    return metrics


# ============================================================
# TRAIN & VALIDATE MODELS
# ============================================================

def train_models(
    models,
    X_train,
    y_train,
    X_validation,
    y_validation,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "🤖 TRAINING MACHINE LEARNING MODELS"
    )

    print(
        "=" * 70
    )

    validation_results = []

    trained_models = {}

    for name, model in models.items():

        print(
            f"\n🚀 Training: {name}"
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        trained_models[
            name
        ] = model

        validation_results.append(
            {
                "Model": name,
                **metrics,
            }
        )

        print(
            f"   MAE  : "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"   RMSE : "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"   R²   : "
            f"{metrics['R2']:.4f}"
        )

        print(
            f"   MAPE : "
            f"{metrics['MAPE_Percent']:.2f}%"
        )

    results_df = pd.DataFrame(
        validation_results
    )

    results_df = (
        results_df
        .sort_values(
            "RMSE",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        trained_models,
        results_df,
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    trained_models,
    validation_results,
):

    best_model_name = (
        validation_results
        .iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "🏆 BEST VALIDATION MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: {best_model_name}"
    )

    best_row = (
        validation_results
        .iloc[0]
    )

    print(
        f"RMSE : "
        f"{best_row['RMSE']:.4f}"
    )

    print(
        f"MAE  : "
        f"{best_row['MAE']:.4f}"
    )

    print(
        f"R²   : "
        f"{best_row['R2']:.4f}"
    )

    print(
        f"MAPE : "
        f"{best_row['MAPE_Percent']:.2f}%"
    )

    return (
        best_model_name,
        best_model,
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_test_set(
    best_model_name,
    best_model,
    X_test,
    y_test,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "🧪 FINAL TEST SET EVALUATION"
    )

    print(
        "=" * 70
    )

    predictions = best_model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print(
        f"\nModel: {best_model_name}"
    )

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    print(
        f"MAPE : {metrics['MAPE_Percent']:.2f}%"
    )

    return (
        predictions,
        metrics,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    validation_results,
    best_model_name,
    test_metrics,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = (
        validation_results
        .copy()
    )

    # --------------------------------------------------------
    # Add final test model information
    # --------------------------------------------------------

    test_row = pd.DataFrame(
        [
            {
                "Model": (
                    best_model_name
                    + " - Final Test"
                ),
                "MAE": test_metrics[
                    "MAE"
                ],
                "RMSE": test_metrics[
                    "RMSE"
                ],
                "R2": test_metrics[
                    "R2"
                ],
                "MAPE_Percent": test_metrics[
                    "MAPE_Percent"
                ],
            }
        ]
    )

    results = pd.concat(
        [
            results,
            test_row,
        ],
        ignore_index=True,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print(
        f"\n💾 Model comparison saved:"
        f"\n   {RESULTS_FILE}"
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
        "BASELINE MACHINE LEARNING PIPELINE"
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
    ) = load_datasets()

    # --------------------------------------------------------
    # Prepare features
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

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    (
        numerical_features,
        categorical_features,
    ) = identify_feature_types(
        X_train
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # Naive baseline
    # --------------------------------------------------------

    naive_metrics = (
        evaluate_naive_baseline(
            y_train,
            y_validation,
        )
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = build_models(
        preprocessor
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    (
        trained_models,
        validation_results,
    ) = train_models(
        models,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    # --------------------------------------------------------
    # Add naive baseline
    # --------------------------------------------------------

    naive_row = pd.DataFrame(
        [
            {
                "Model": "Naive Baseline",
                **naive_metrics,
            }
        ]
    )

    validation_results = pd.concat(
        [
            naive_row,
            validation_results,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    validation_results = (
        validation_results
        .sort_values(
            "RMSE",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 MODEL COMPARISON — VALIDATION SET"
    )

    print(
        "=" * 70
    )

    print(
        validation_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select best ML model
    # --------------------------------------------------------

    ml_results = (
        validation_results[
            validation_results["Model"]
            != "Naive Baseline"
        ]
        .reset_index(
            drop=True
        )
    )

    best_model_name = (
        ml_results
        .iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print(
        "\n🏆 Selected model:"
    )

    print(
        f"   {best_model_name}"
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    (
        test_predictions,
        test_metrics,
    ) = evaluate_test_set(
        best_model_name,
        best_model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        validation_results,
        best_model_name,
        test_metrics,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ STEP 7 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\n🎯 Target:"
    )

    print(
        f"   {TARGET}"
    )

    print(
        "\n📈 Models evaluated:"
    )

    print(
        "   • Naive Baseline"
    )

    print(
        "   • Linear Regression"
    )

    print(
        "   • Random Forest"
    )

    print(
        "   • Gradient Boosting"
    )

    print(
        "\n🏆 Best model:"
    )

    print(
        f"   {best_model_name}"
    )

    print(
        "\n🧪 Final test performance:"
    )

    print(
        f"   MAE  : {test_metrics['MAE']:.4f}"
    )

    print(
        f"   RMSE : {test_metrics['RMSE']:.4f}"
    )

    print(
        f"   R²   : {test_metrics['R2']:.4f}"
    )

    print(
        f"   MAPE : {test_metrics['MAPE_Percent']:.2f}%"
    )

    print(
        "\n📁 Results:"
    )

    print(
        f"   {RESULTS_FILE}"
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
            "\n❌ MODEL TRAINING FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)