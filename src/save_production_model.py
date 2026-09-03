"""
Healthcare Medical-Device Sales ML Pipeline
Step 11.1: Production Model Packaging

Purpose
-------
Train the finalized Gradient Boosting model on the approved
training + validation data and save the complete preprocessing
+ model pipeline as a reusable production artifact.

Important
---------
The test set is NOT used for training.

The test set remains untouched so that the previously reported
final test performance remains an unbiased evaluation.
"""

from pathlib import Path
import sys
import json
import warnings

import pandas as pd
import joblib

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

MODEL_FILE = (
    OUTPUT_DIR
    / "medical_device_demand_model.joblib"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "model_metadata.json"
)


# ============================================================
# TARGET VARIABLE
# ============================================================

TARGET = "Units_Sold"


# ============================================================
# FINAL MODEL PARAMETERS
# ============================================================

BEST_PARAMETERS = {

    "n_estimators": 300,

    "learning_rate": 0.03,

    "max_depth": 7,

    "min_samples_leaf": 3,

}


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
        "\n📂 Loading datasets..."
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
        f"Training rows   : {len(train_df):,}"
    )

    print(
        f"Validation rows : {len(validation_df):,}"
    )

    print(
        f"Test rows       : {len(test_df):,}"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# COMBINE TRAINING + VALIDATION
# ============================================================

def combine_training_data(
    train_df,
    validation_df,
):

    print(
        "\n🔗 Combining training and validation data..."
    )

    combined_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        axis=0,
        ignore_index=True,
    )

    print(
        f"Final production-training rows: "
        f"{len(combined_df):,}"
    )

    return combined_df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    combined_df,
    test_df,
):

    feature_columns = [

        column

        for column in combined_df.columns

        if column not in EXCLUDED_COLUMNS

    ]

    feature_columns = [

        column

        for column in feature_columns

        if column in test_df.columns

    ]

    X = (
        combined_df[
            feature_columns
        ]
        .copy()
    )

    y = (
        combined_df[TARGET]
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
        f"\n🔢 Production model features: "
        f"{len(feature_columns)}"
    )

    return (
        X,
        y,
        X_test,
        y_test,
        feature_columns,
    )


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(
    X,
):

    numerical_features = (

        X

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

        X

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
# BUILD FINAL MODEL
# ============================================================

def build_model(
    preprocessor,
):

    print(
        "\n🤖 Building optimized Gradient "
        "Boosting model..."
    )

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
# TRAIN PRODUCTION MODEL
# ============================================================

def train_production_model(
    pipeline,
    X,
    y,
):

    print(
        "\n🏋️ Training production model..."
    )

    print(
        "Training data = Train + Validation"
    )

    pipeline.fit(
        X,
        y,
    )

    print(
        "✅ Production model trained."
    )

    return pipeline


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    pipeline,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_FILE,
    )

    print(
        f"\n💾 Model saved:"
    )

    print(
        f"   {MODEL_FILE}"
    )


# ============================================================
# CREATE MODEL METADATA
# ============================================================

def create_metadata(
    feature_columns,
    train_rows,
    validation_rows,
    test_rows,
):

    metadata = {

        "project": (
            "Healthcare Medical-Device "
            "Sales Demand Prediction"
        ),

        "target": TARGET,

        "model": (
            "GradientBoostingRegressor"
        ),

        "model_parameters":
            BEST_PARAMETERS,

        "features": feature_columns,

        "number_of_features":
            len(feature_columns),

        "training_rows":
            train_rows,

        "validation_rows":
            validation_rows,

        "test_rows":
            test_rows,

        "training_strategy":
            "Train + Validation combined "
            "after model selection",

        "test_strategy":
            "Test set retained separately "
            "for final evaluation",

        "random_state":
            42,

        "artifact":
            MODEL_FILE.name,

    }

    return metadata


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    metadata,
):

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"\n💾 Metadata saved:"
    )

    print(
        f"   {METADATA_FILE}"
    )


# ============================================================
# VERIFY MODEL ARTIFACT
# ============================================================

def verify_model_artifact():

    print(
        "\n🔍 Verifying saved model artifact..."
    )

    loaded_model = joblib.load(
        MODEL_FILE
    )

    if loaded_model is None:

        raise RuntimeError(
            "Saved model artifact is empty."
        )

    print(
        "✅ Model artifact loaded successfully."
    )

    return loaded_model


# ============================================================
# VERIFY PREDICTION
# ============================================================

def verify_prediction(
    loaded_model,
    X_test,
):

    print(
        "\n🧪 Running production-artifact "
        "prediction test..."
    )

    sample_size = min(
        5,
        len(X_test),
    )

    sample = (
        X_test
        .head(sample_size)
        .copy()
    )

    predictions = (
        loaded_model.predict(
            sample
        )
    )

    if len(predictions) != sample_size:

        raise RuntimeError(
            "Prediction verification failed."
        )

    print(
        "\nSample predictions:"
    )

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):

        print(
            f"   Sample {index}: "
            f"{prediction:.4f} units"
        )

    print(
        "\n✅ Prediction verification passed."
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
        "STEP 11.1 — PRODUCTION MODEL PACKAGING"
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
    # Combine approved training data
    # --------------------------------------------------------

    combined_df = (
        combine_training_data(
            train_df,
            validation_df,
        )
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    (
        X,
        y,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_features(
        combined_df,
        test_df,
    )

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    (
        numerical_features,
        categorical_features,
    ) = identify_feature_types(
        X
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
    # Model
    # --------------------------------------------------------

    pipeline = build_model(
        preprocessor
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline = (
        train_production_model(
            pipeline,
            X,
            y,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        pipeline
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = create_metadata(

        feature_columns,

        len(train_df),

        len(validation_df),

        len(test_df),

    )

    save_metadata(
        metadata
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    loaded_model = (
        verify_model_artifact()
    )

    verify_prediction(
        loaded_model,
        X_test,
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ STEP 11.1 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\n📦 Production artifacts:"
    )

    print(
        f"   • {MODEL_FILE.name}"
    )

    print(
        f"   • {METADATA_FILE.name}"
    )

    print(
        "\n🚀 The trained ML pipeline is now "
        "packaged for deployment."
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
            "\n❌ PRODUCTION MODEL PACKAGING FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)