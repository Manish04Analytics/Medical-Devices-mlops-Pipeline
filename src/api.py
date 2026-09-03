"""
Medical Device Demand Prediction API
=====================================

Step 11.3 / 11.4
Production-style FastAPI service for the trained
medical-device demand prediction pipeline.

The API:
- Loads the production model
- Uses the exact 49 model features
- Validates incoming requests
- Preserves the exact feature order
- Generates demand predictions
- Provides health and model-information endpoints
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "medical_device_demand_model.joblib"
)


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Production model not found:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# GET EXACT MODEL FEATURES
# ============================================================

if not hasattr(model, "feature_names_in_"):
    raise RuntimeError(
        "The saved production model does not contain "
        "'feature_names_in_'."
    )

MODEL_FEATURES = list(
    model.feature_names_in_
)


# ============================================================
# VERIFY EXPECTED FEATURE COUNT
# ============================================================

if len(MODEL_FEATURES) != 49:
    raise RuntimeError(
        "Unexpected production model feature count. "
        f"Expected 49, found {len(MODEL_FEATURES)}."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Medical Device Demand Prediction API",
    description=(
        "Production-style machine learning API for "
        "medical-device demand prediction."
    ),
    version="1.0.0",
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    # --------------------------------------------------------
    # Categorical Features
    # --------------------------------------------------------

    Hospital_Type: str

    Hospital_Size: str

    Territory: str

    Product_Category: str

    # --------------------------------------------------------
    # Transaction / Operational Features
    # --------------------------------------------------------

    Unit_Price: float = Field(
        ...,
        ge=0
    )

    Inventory_Level: int = Field(
        ...,
        ge=0
    )

    Stockout_Flag: int = Field(
        ...,
        ge=0,
        le=1
    )

    Market_Index: float = Field(
        ...,
        ge=0,
        le=1
    )

    Promotion_Flag: int = Field(
        ...,
        ge=0,
        le=1
    )

    # --------------------------------------------------------
    # Calendar Features
    # --------------------------------------------------------

    Year: int = Field(
        ...,
        ge=2000
    )

    Month: int = Field(
        ...,
        ge=1,
        le=12
    )

    Quarter: int = Field(
        ...,
        ge=1,
        le=4
    )

    Day_of_Week: int = Field(
        ...,
        ge=0,
        le=6
    )

    Is_Weekend: int = Field(
        ...,
        ge=0,
        le=1
    )

    Day_of_Month: int = Field(
        ...,
        ge=1,
        le=31
    )

    Week_of_Year: int = Field(
        ...,
        ge=1,
        le=53
    )

    # --------------------------------------------------------
    # Aggregated Features
    # --------------------------------------------------------

    Average_Unit_Price: float = Field(
        ...,
        ge=0
    )

    Average_Inventory: float = Field(
        ...,
        ge=0
    )

    Stockout_Rate: float = Field(
        ...,
        ge=0,
        le=1
    )

    Promotion_Rate: float = Field(
        ...,
        ge=0,
        le=1
    )

    Average_Market_Index: float = Field(
        ...,
        ge=0,
        le=1
    )

    # --------------------------------------------------------
    # Demand Lag Features
    # --------------------------------------------------------

    Demand_Lag_1: float = Field(
        ...,
        ge=0
    )

    Demand_Lag_2: float = Field(
        ...,
        ge=0
    )

    Demand_Lag_3: float = Field(
        ...,
        ge=0
    )

    Demand_Lag_7: float = Field(
        ...,
        ge=0
    )

    Demand_Lag_14: float = Field(
        ...,
        ge=0
    )

    Demand_Lag_28: float = Field(
        ...,
        ge=0
    )

    # --------------------------------------------------------
    # Rolling Demand Features
    # --------------------------------------------------------

    Demand_Rolling_Mean_3: float = Field(
        ...,
        ge=0
    )

    Demand_Rolling_Mean_7: float = Field(
        ...,
        ge=0
    )

    Demand_Rolling_Mean_14: float = Field(
        ...,
        ge=0
    )

    Demand_Rolling_Mean_28: float = Field(
        ...,
        ge=0
    )

    # --------------------------------------------------------
    # Rolling Standard Deviation Features
    # --------------------------------------------------------

    Demand_Rolling_Std_3: float = Field(
        ...,
        ge=0
    )

    Demand_Rolling_Std_7: float = Field(
        ...,
        ge=0
    )

    Demand_Rolling_Std_14: float = Field(
        ...,
        ge=0
    )

    Demand_Rolling_Std_28: float = Field(
        ...,
        ge=0
    )

    # --------------------------------------------------------
    # Demand Trend Features
    # --------------------------------------------------------

    Demand_Growth_7D: float

    Demand_Growth_28D: float

    Demand_Acceleration: float

    # --------------------------------------------------------
    # Inventory Risk Features
    # --------------------------------------------------------

    Inventory_Coverage: float = Field(
        ...,
        ge=0
    )

    Inventory_Pressure: float

    Low_Inventory_Flag: int = Field(
        ...,
        ge=0,
        le=1
    )

    High_Inventory_Flag: int = Field(
        ...,
        ge=0,
        le=1
    )

    Stockout_Risk_Signal: int = Field(
        ...,
        ge=0,
        le=1
    )

    # --------------------------------------------------------
    # Promotion Features
    # --------------------------------------------------------

    Promotion_Lag_1: float = Field(
        ...,
        ge=0
    )

    Promotion_Lag_7: float = Field(
        ...,
        ge=0
    )

    Promotion_Active: int = Field(
        ...,
        ge=0,
        le=1
    )

    # IMPORTANT:
    # This is a frequency/count feature.
    # It is NOT restricted to 0-1.
    #
    # Example valid values:
    # 0.0
    # 1.0
    # 2.0
    # 3.0
    # etc.

    Promotion_7D_Frequency: float = Field(
        ...,
        ge=0
    )

    # --------------------------------------------------------
    # Volatility Features
    # --------------------------------------------------------

    Demand_CV_28D: float = Field(
        ...,
        ge=0
    )

    Demand_Volatility_Flag: int = Field(
        ...,
        ge=0,
        le=1
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:

    return {

        "project":
            "Medical Device Demand Prediction",

        "status":
            "running",

        "model":
            "Gradient Boosting Regressor",

        "model_version":
            "1.0.0",

        "features_required":
            len(MODEL_FEATURES),

        "documentation":
            "/docs",

        "health_check":
            "/health",

        "model_information":
            "/model-info",

        "prediction_endpoint":
            "/predict",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check() -> dict[str, Any]:

    return {

        "status":
            "healthy",

        "model_loaded":
            model is not None,

        "model_file":
            MODEL_PATH.name,

        "features_required":
            len(MODEL_FEATURES),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info() -> dict[str, Any]:

    return {

        "model_type":
            "GradientBoostingRegressor",

        "model_version":
            "1.0.0",

        "feature_count":
            len(MODEL_FEATURES),

        "features":
            MODEL_FEATURES,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
) -> dict[str, Any]:

    try:

        # ----------------------------------------------------
        # Convert Pydantic request to dictionary
        # ----------------------------------------------------

        input_data = request.model_dump()

        # ----------------------------------------------------
        # Check for missing model features
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in MODEL_FEATURES

            if feature not in input_data

        ]

        if missing_features:

            raise HTTPException(

                status_code=422,

                detail={
                    "error":
                        "Missing model features",

                    "missing_features":
                        missing_features,
                },
            )

        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # Force exact production feature order
        # ----------------------------------------------------

        input_df = input_df[
            MODEL_FEATURES
        ]

        # ----------------------------------------------------
        # Final feature-order validation
        # ----------------------------------------------------

        if list(
            input_df.columns
        ) != MODEL_FEATURES:

            raise HTTPException(

                status_code=500,

                detail=(
                    "Feature ordering mismatch "
                    "before prediction."
                ),
            )

        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )

        predicted_units = float(
            prediction[0]
        )

        # ----------------------------------------------------
        # Demand cannot be negative
        # ----------------------------------------------------

        predicted_units = max(
            0.0,
            predicted_units
        )

        # ----------------------------------------------------
        # Return production response
        # ----------------------------------------------------

        return {

            "status":
                "success",

            "prediction": {

                "predicted_units_sold":
                    round(
                        predicted_units,
                        2
                    ),

                "unit":
                    "units",
            },

            "model": {

                "name":
                    "Gradient Boosting Regressor",

                "version":
                    "1.0.0",

                "features_used":
                    len(MODEL_FEATURES),
            },
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail={

                "error":
                    "Prediction failed",

                "message":
                    str(exc),
            },
        )


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
def startup_event():

    print(
        "\n" + "=" * 70
    )

    print(
        "🏥 MEDICAL DEVICE DEMAND "
        "PREDICTION API"
    )

    print(
        "=" * 70
    )

    print(
        "✅ Production model loaded."
    )

    print(
        f"📦 Model: {MODEL_PATH.name}"
    )

    print(
        f"🔢 Required features: "
        f"{len(MODEL_FEATURES)}"
    )

    print(
        "🔒 Exact feature order enforced."
    )

    print(
        "🌐 API ready."
    )

    print(
        "📖 Swagger documentation:"
    )

    print(
        "   http://127.0.0.1:8000/docs"
    )

    print(
        "=" * 70
    )