"""
Healthcare Medical-Device Sales
Step 9: Gradient Boosting Hyperparameter Optimization

Purpose
-------
Systematically optimize the Gradient Boosting model while
preserving the chronological train/validation/test design.

Important
---------
The test set is NOT used for hyperparameter selection.

Training data:
    Used to fit candidate models.

Validation data:
    Used to select the best hyperparameters.

Test data:
    Used only once for final evaluation.
"""

from pathlib import Path
import sys
import warnings
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

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
    DATA_DIR / "train.csv"
)

VALIDATION_FILE = (
    DATA_DIR / "validation.csv"
)

TEST_FILE = (
    DATA_DIR / "test.csv"
)

TUNING_RESULTS_FILE = (
    OUTPUT_DIR
    / "hyperparameter_tuning_results.csv"
)

BEST_PARAMS_FILE = (
    OUTPUT_DIR
    / "best_model_parameters.json"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Units_Sold"


# ============================================================
# EXCLUDED FEATURES
# ============================================================

EXCLUDED_COLUMNS = [

    # Target
    "Units_Sold",

    # Raw date
    "Transaction_Date",

    # Identifiers
    "Hospital_ID",
    "Hospital_Name",
    "Product_ID",
    "Product_Name",

    # Direct target-derived variables
    "Total_Revenue",
    "Calculated_Revenue",
    "Revenue_Difference",

    "Daily_Units_Sold",
    "Daily_Revenue",

    # Potential aggregate leakage
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
        f"✅ Training rows   : {len(train_df):,}"
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

        if (
            column in validation_df.columns
            and
            column in test_df.columns
        )

    ]

    X_train = (
        train_df[
            feature_columns
        ]
        .copy()
    )

    y_train = (
        train_df[TARGET]
        .copy()
    )

    X_validation = (
        validation_df[
            feature_columns
        ]
        .copy()
    )

    y_validation = (
        validation_df[TARGET]
        .copy()
    )

    X_test = (
        test_df[
            feature_columns
        ]
        .copy()
    )

    y_test = (
        test_df[TARGET]
        .copy()
    )

    print(
        f"\n🔢 Features used: "
        f"{len(feature_columns)}"
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# FEATURE TYPES
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
        f"🔢 Numerical features   : "
        f"{len(numerical_features)}"
    )

    print(
        f"🔤 Categorical features : "
        f"{len(categorical_features)}"
    )

    return (
        numerical_features,
        categorical_features,
    )


# ============================================================
# PREPROCESSOR
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

    y_true_array = (
        np.asarray(y_true)
    )

    predictions_array = (
        np.asarray(predictions)
    )

    non_zero_mask = (
        y_true_array != 0
    )

    if non_zero_mask.sum() > 0:

        mape = (

            np.mean(

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

            )

            * 100

        )

    else:

        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_Percent": mape,
    }


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

def create_parameter_grid():

    parameter_grid = [

        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_samples_leaf": 2,
        },

        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_samples_leaf": 5,
        },

        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_samples_leaf": 5,
        },

        {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_samples_leaf": 5,
        },

        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 7,
            "min_samples_leaf": 3,
        },

    ]

    return parameter_grid


# ============================================================
# TUNING
# ============================================================

def run_hyperparameter_search(
    X_train,
    y_train,
    X_validation,
    y_validation,
    preprocessor,
):

    parameter_grid = (
        create_parameter_grid()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "🔧 HYPERPARAMETER OPTIMIZATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCandidate configurations: "
        f"{len(parameter_grid)}"
    )

    results = []

    best_rmse = float("inf")

    best_parameters = None

    best_pipeline = None

    for index, params in enumerate(
        parameter_grid,
        start=1,
    ):

        print(
            f"\n[{index}/"
            f"{len(parameter_grid)}]"
            f" Testing: {params}"
        )

        model = (
            GradientBoostingRegressor(
                n_estimators=params[
                    "n_estimators"
                ],

                learning_rate=params[
                    "learning_rate"
                ],

                max_depth=params[
                    "max_depth"
                ],

                min_samples_leaf=params[
                    "min_samples_leaf"
                ],

                random_state=42,
            )
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

        predictions = (
            pipeline.predict(
                X_validation
            )
        )

        metrics = (
            calculate_metrics(
                y_validation,
                predictions,
            )
        )

        result = {
            **params,
            **metrics,
        }

        results.append(
            result
        )

        print(
            f"   RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"   MAE : "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"   R²  : "
            f"{metrics['R2']:.4f}"
        )

        if (
            metrics["RMSE"]
            < best_rmse
        ):

            best_rmse = (
                metrics["RMSE"]
            )

            best_parameters = (
                params.copy()
            )

            best_pipeline = (
                pipeline
            )

            print(
                "   🏆 NEW BEST MODEL"
            )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "RMSE",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        results_df,
        best_parameters,
        best_pipeline,
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results_df,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 HYPERPARAMETER SEARCH RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\n🏆 TOP 5 CONFIGURATIONS"
    )

    print(
        results_df
        .head(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    best_parameters,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        TUNING_RESULTS_FILE,
        index=False,
    )

    with open(
        BEST_PARAMS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_parameters,
            file,
            indent=4,
        )

    print(
        f"\n💾 Tuning results saved:"
        f"\n   {TUNING_RESULTS_FILE}"
    )

    print(
        f"\n💾 Best parameters saved:"
        f"\n   {BEST_PARAMS_FILE}"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_final_model(
    best_pipeline,
    X_test,
    y_test,
    best_parameters,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "🧪 FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    predictions = (
        best_pipeline.predict(
            X_test
        )
    )

    metrics = (
        calculate_metrics(
            y_test,
            predictions,
        )
    )

    print(
        "\nBest hyperparameters:"
    )

    for key, value in (
        best_parameters.items()
    ):

        print(
            f"   {key}: {value}"
        )

    print(
        "\nFinal Test Performance:"
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

    return metrics


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
        "STEP 9 — MODEL OPTIMIZATION"
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
    # Features
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
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
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = (
        build_preprocessor(
            numerical_features,
            categorical_features,
        )
    )

    # --------------------------------------------------------
    # Hyperparameter optimization
    # --------------------------------------------------------

    (
        results_df,
        best_parameters,
        best_pipeline,
    ) = run_hyperparameter_search(
        X_train,
        y_train,
        X_validation,
        y_validation,
        preprocessor,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        results_df
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    final_metrics = (
        evaluate_final_model(
            best_pipeline,
            X_test,
            y_test,
            best_parameters,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results_df,
        best_parameters,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ STEP 9 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\n🏆 Optimized model:"
    )

    print(
        "   Gradient Boosting Regressor"
    )

    print(
        "\n📈 Final test metrics:"
    )

    print(
        f"   MAE  : "
        f"{final_metrics['MAE']:.4f}"
    )

    print(
        f"   RMSE : "
        f"{final_metrics['RMSE']:.4f}"
    )

    print(
        f"   R²   : "
        f"{final_metrics['R2']:.4f}"
    )

    print(
        f"   MAPE : "
        f"{final_metrics['MAPE_Percent']:.2f}%"
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
            "\n❌ MODEL TUNING FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)