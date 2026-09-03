"""
Healthcare Medical-Device Sales
Step 10: Final Model Prediction Diagnostics

Purpose
-------
Evaluate the optimized Gradient Boosting model at the
individual prediction level.

Outputs
-------
outputs/
    final_test_predictions.csv
    actual_vs_predicted.png
    residual_analysis.png
    residual_distribution.png
    prediction_error_analysis.csv
    final_model_diagnostics.txt

Important
---------
The test dataset is used ONLY for final evaluation.
No hyperparameter tuning is performed in this script.
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

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "final_test_predictions.csv"
)

ACTUAL_PREDICTED_PLOT = (
    OUTPUT_DIR
    / "actual_vs_predicted.png"
)

RESIDUAL_PLOT = (
    OUTPUT_DIR
    / "residual_analysis.png"
)

RESIDUAL_DISTRIBUTION_PLOT = (
    OUTPUT_DIR
    / "residual_distribution.png"
)

ERROR_ANALYSIS_FILE = (
    OUTPUT_DIR
    / "prediction_error_analysis.csv"
)

DIAGNOSTICS_REPORT = (
    OUTPUT_DIR
    / "final_model_diagnostics.txt"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Units_Sold"


# ============================================================
# OPTIMIZED MODEL PARAMETERS
# ============================================================

BEST_PARAMETERS = {

    "n_estimators": 300,

    "learning_rate": 0.03,

    "max_depth": 7,

    "min_samples_leaf": 3,

}


# ============================================================
# EXCLUDED COLUMNS
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

    # Aggregate demand variables
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
        f"\n🔢 Model features: "
        f"{len(feature_columns)}"
    )

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
# BUILD OPTIMIZED MODEL
# ============================================================

def build_model(
    preprocessor,
):

    model = GradientBoostingRegressor(

        n_estimators=(
            BEST_PARAMETERS[
                "n_estimators"
            ]
        ),

        learning_rate=(
            BEST_PARAMETERS[
                "learning_rate"
            ]
        ),

        max_depth=(
            BEST_PARAMETERS[
                "max_depth"
            ]
        ),

        min_samples_leaf=(
            BEST_PARAMETERS[
                "min_samples_leaf"
            ]
        ),

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

    return pipeline


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    pipeline,
    X_train,
    y_train,
):

    print(
        "\n🤖 Training optimized Gradient "
        "Boosting model..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "✅ Final model trained."
    )

    return pipeline


# ============================================================
# CALCULATE METRICS
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
# CREATE PREDICTION DATASET
# ============================================================

def create_prediction_dataset(
    test_df,
    predictions,
):

    results = test_df.copy()

    results[
        "Actual_Units_Sold"
    ] = results[
        TARGET
    ]

    results[
        "Predicted_Units_Sold"
    ] = predictions

    results[
        "Prediction_Error"
    ] = (
        results[
            "Actual_Units_Sold"
        ]

        -

        results[
            "Predicted_Units_Sold"
        ]
    )

    results[
        "Absolute_Error"
    ] = np.abs(
        results[
            "Prediction_Error"
        ]
    )

    results[
        "Absolute_Percentage_Error"
    ] = np.where(

        results[
            "Actual_Units_Sold"
        ] != 0,

        (

            results[
                "Absolute_Error"
            ]

            /

            results[
                "Actual_Units_Sold"
            ]

        )

        * 100,

        np.nan,

    )

    results[
        "Prediction_Direction"
    ] = np.where(

        results[
            "Prediction_Error"
        ] > 0,

        "Underprediction",

        np.where(

            results[
                "Prediction_Error"
            ] < 0,

            "Overprediction",

            "Exact",

        ),

    )

    return results


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

def create_actual_vs_predicted_plot(
    y_test,
    predictions,
):

    print(
        "\n📊 Creating actual-vs-predicted plot..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6,
    )

    minimum = min(
        np.min(y_test),
        np.min(predictions),
    )

    maximum = max(
        np.max(y_test),
        np.max(predictions),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Actual Units Sold"
    )

    plt.ylabel(
        "Predicted Units Sold"
    )

    plt.title(
        "Actual vs Predicted Medical-Device Demand"
    )

    plt.tight_layout()

    plt.savefig(
        ACTUAL_PREDICTED_PLOT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"✅ Saved: "
        f"{ACTUAL_PREDICTED_PLOT}"
    )


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

def create_residual_plot(
    predictions,
    residuals,
):

    print(
        "\n📉 Creating residual analysis plot..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.scatter(
        predictions,
        residuals,
        alpha=0.6,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel(
        "Predicted Units Sold"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Analysis"
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_PLOT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"✅ Saved: "
        f"{RESIDUAL_PLOT}"
    )


# ============================================================
# RESIDUAL DISTRIBUTION
# ============================================================

def create_residual_distribution(
    residuals,
):

    print(
        "\n📊 Creating residual distribution..."
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.hist(
        residuals,
        bins=30,
    )

    plt.axvline(
        x=0,
        linestyle="--",
    )

    plt.xlabel(
        "Prediction Residual"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Distribution of Prediction Residuals"
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_DISTRIBUTION_PLOT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"✅ Saved: "
        f"{RESIDUAL_DISTRIBUTION_PLOT}"
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def perform_error_analysis(
    prediction_df,
):

    print(
        "\n🔍 Performing prediction-error analysis..."
    )

    analysis = prediction_df[
        [
            "Actual_Units_Sold",
            "Predicted_Units_Sold",
            "Prediction_Error",
            "Absolute_Error",
            "Absolute_Percentage_Error",
            "Prediction_Direction",
        ]
    ].copy()

    summary = (

        analysis

        .groupby(
            "Prediction_Direction"
        )

        .agg(

            Predictions=(
                "Prediction_Error",
                "count",
            ),

            Mean_Absolute_Error=(
                "Absolute_Error",
                "mean",
            ),

            Median_Absolute_Error=(
                "Absolute_Error",
                "median",
            ),

            Mean_Percentage_Error=(
                "Absolute_Percentage_Error",
                "mean",
            ),

        )

        .reset_index()

    )

    print(
        "\n📋 Error summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    prediction_df,
):

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"\n💾 Predictions saved:"
        f"\n   {PREDICTIONS_FILE}"
    )


# ============================================================
# SAVE ERROR ANALYSIS
# ============================================================

def save_error_analysis(
    summary,
):

    summary.to_csv(
        ERROR_ANALYSIS_FILE,
        index=False,
    )

    print(
        f"\n💾 Error analysis saved:"
        f"\n   {ERROR_ANALYSIS_FILE}"
    )


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    metrics,
    prediction_df,
    summary,
):

    residuals = (
        prediction_df[
            "Prediction_Error"
        ]
    )

    report = []

    report.append(
        "HEALTHCARE MEDICAL-DEVICE SALES"
    )

    report.append(
        "FINAL MODEL DIAGNOSTICS REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append("")

    report.append(
        "MODEL"
    )

    report.append(
        "Optimized Gradient Boosting Regressor"
    )

    report.append("")

    report.append(
        "HYPERPARAMETERS"
    )

    report.append(
        "-" * 70
    )

    for key, value in (
        BEST_PARAMETERS.items()
    ):

        report.append(
            f"{key}: {value}"
        )

    report.append("")

    report.append(
        "FINAL TEST PERFORMANCE"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    report.append(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    report.append(
        f"R²   : {metrics['R2']:.4f}"
    )

    report.append(
        f"MAPE : {metrics['MAPE_Percent']:.2f}%"
    )

    report.append("")

    report.append(
        "PREDICTION DIAGNOSTICS"
    )

    report.append(
        "-" * 70
    )

    report.append(
        f"Number of test predictions: "
        f"{len(prediction_df):,}"
    )

    report.append(
        f"Mean residual: "
        f"{residuals.mean():.4f}"
    )

    report.append(
        f"Median residual: "
        f"{residuals.median():.4f}"
    )

    report.append(
        f"Residual standard deviation: "
        f"{residuals.std():.4f}"
    )

    report.append(
        f"Maximum absolute error: "
        f"{residuals.abs().max():.4f}"
    )

    report.append("")

    report.append(
        "PREDICTION DIRECTION"
    )

    report.append(
        "-" * 70
    )

    direction_counts = (
        prediction_df[
            "Prediction_Direction"
        ]
        .value_counts()
    )

    for direction, count in (
        direction_counts.items()
    ):

        percentage = (
            count
            /
            len(prediction_df)
            * 100
        )

        report.append(
            f"{direction}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    report.append("")

    report.append(
        "INTERPRETATION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "The final test set was kept separate "
        "from hyperparameter selection."
    )

    report.append(
        "Residuals represent the difference between "
        "actual and predicted demand."
    )

    report.append(
        "A residual near zero indicates a prediction "
        "close to the observed demand."
    )

    report.append(
        "Residual analysis should be used to identify "
        "systematic prediction errors and potential "
        "areas for future model improvement."
    )

    report.append("")

    report.append(
        "CAUTION"
    )

    report.append(
        "-" * 70
    )

    report.append(
        "Predictive performance does not establish "
        "causal relationships."
    )

    report.append(
        "The model should support demand-planning "
        "decisions rather than replace human review."
    )

    with open(
        DIAGNOSTICS_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(report)
        )

    print(
        f"\n📝 Diagnostics report saved:"
        f"\n   {DIAGNOSTICS_REPORT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "🏥 HEALTHCARE MEDICAL-DEVICE SALES"
    )

    print(
        "STEP 10 — FINAL MODEL DIAGNOSTICS"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    preprocessor = (
        build_preprocessor(
            numerical_features,
            categorical_features,
        )
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    pipeline = build_model(
        preprocessor
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline = train_final_model(
        pipeline,
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "\n🔮 Generating test-set predictions..."
    )

    predictions = (
        pipeline.predict(
            X_test
        )
    )

    print(
        "✅ Predictions generated."
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = (
        calculate_metrics(
            y_test,
            predictions,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "📈 FINAL TEST PERFORMANCE"
    )

    print(
        "=" * 70
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

    # --------------------------------------------------------
    # Prediction dataset
    # --------------------------------------------------------

    prediction_df = (
        create_prediction_dataset(
            test_df,
            predictions,
        )
    )

    # --------------------------------------------------------
    # Residuals
    # --------------------------------------------------------

    residuals = (
        prediction_df[
            "Prediction_Error"
        ]
        .values
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    create_actual_vs_predicted_plot(
        y_test,
        predictions,
    )

    create_residual_plot(
        predictions,
        residuals,
    )

    create_residual_distribution(
        residuals,
    )

    # --------------------------------------------------------
    # Error analysis
    # --------------------------------------------------------

    summary = (
        perform_error_analysis(
            prediction_df
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_predictions(
        prediction_df
    )

    save_error_analysis(
        summary
    )

    create_report(
        metrics,
        prediction_df,
        summary,
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ STEP 10 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\n📁 Generated outputs:"
    )

    print(
        f"   • {PREDICTIONS_FILE.name}"
    )

    print(
        f"   • {ACTUAL_PREDICTED_PLOT.name}"
    )

    print(
        f"   • {RESIDUAL_PLOT.name}"
    )

    print(
        f"   • {RESIDUAL_DISTRIBUTION_PLOT.name}"
    )

    print(
        f"   • {ERROR_ANALYSIS_FILE.name}"
    )

    print(
        f"   • {DIAGNOSTICS_REPORT.name}"
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
            "\n❌ FINAL MODEL DIAGNOSTICS FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)