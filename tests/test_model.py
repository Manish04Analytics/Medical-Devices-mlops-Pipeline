"""
Automated Tests — Production Medical Device Demand Model
==========================================================

Step 12.2

Tests:
1. Production model file exists
2. Model loads successfully
3. Model contains exactly 49 features
4. Expected feature names are present
5. Model can generate a prediction
6. Prediction is numeric and non-negative
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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
# EXPECTED MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "Hospital_Type",
    "Hospital_Size",
    "Territory",
    "Product_Category",
    "Unit_Price",
    "Inventory_Level",
    "Stockout_Flag",
    "Market_Index",
    "Promotion_Flag",
    "Year",
    "Month",
    "Quarter",
    "Day_of_Week",
    "Is_Weekend",
    "Day_of_Month",
    "Week_of_Year",
    "Average_Unit_Price",
    "Average_Inventory",
    "Stockout_Rate",
    "Promotion_Rate",
    "Average_Market_Index",
    "Demand_Lag_1",
    "Demand_Lag_2",
    "Demand_Lag_3",
    "Demand_Lag_7",
    "Demand_Lag_14",
    "Demand_Lag_28",
    "Demand_Rolling_Mean_3",
    "Demand_Rolling_Mean_7",
    "Demand_Rolling_Mean_14",
    "Demand_Rolling_Mean_28",
    "Demand_Rolling_Std_3",
    "Demand_Rolling_Std_7",
    "Demand_Rolling_Std_14",
    "Demand_Rolling_Std_28",
    "Demand_Growth_7D",
    "Demand_Growth_28D",
    "Demand_Acceleration",
    "Inventory_Coverage",
    "Inventory_Pressure",
    "Low_Inventory_Flag",
    "High_Inventory_Flag",
    "Stockout_Risk_Signal",
    "Promotion_Lag_1",
    "Promotion_Lag_7",
    "Promotion_Active",
    "Promotion_7D_Frequency",
    "Demand_CV_28D",
    "Demand_Volatility_Flag",
]


# ============================================================
# TEST 1 — MODEL FILE EXISTS
# ============================================================

def test_model_file_exists():

    assert MODEL_PATH.exists(), (
        f"Production model not found: {MODEL_PATH}"
    )

    assert MODEL_PATH.is_file(), (
        f"Model path is not a file: {MODEL_PATH}"
    )


# ============================================================
# TEST 2 — MODEL LOADS SUCCESSFULLY
# ============================================================

def test_model_loads():

    model = joblib.load(MODEL_PATH)

    assert model is not None


# ============================================================
# TEST 3 — MODEL HAS FEATURE NAMES
# ============================================================

def test_model_has_feature_names():

    model = joblib.load(MODEL_PATH)

    assert hasattr(
        model,
        "feature_names_in_"
    ), (
        "Production model does not contain "
        "'feature_names_in_'."
    )


# ============================================================
# TEST 4 — EXACTLY 49 FEATURES
# ============================================================

def test_model_feature_count():

    model = joblib.load(MODEL_PATH)

    actual_features = list(
        model.feature_names_in_
    )

    assert len(actual_features) == 49, (
        f"Expected 49 features, "
        f"found {len(actual_features)}."
    )


# ============================================================
# TEST 5 — FEATURE NAMES MATCH
# ============================================================

def test_model_features_match_expected():

    model = joblib.load(MODEL_PATH)

    actual_features = list(
        model.feature_names_in_
    )

    assert actual_features == EXPECTED_FEATURES, (
        "Production model features do not match "
        "the expected 49-feature schema.\n\n"
        f"Expected:\n{EXPECTED_FEATURES}\n\n"
        f"Actual:\n{actual_features}"
    )


# ============================================================
# TEST 6 — MODEL CAN GENERATE A PREDICTION
# ============================================================

def test_model_prediction():

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Example production observation
    # --------------------------------------------------------

    sample = {
        "Hospital_Type": "Multi_Specialty",
        "Hospital_Size": "Medium",
        "Territory": "Central",
        "Product_Category": "Radiology",

        "Unit_Price": 1857602.99,
        "Inventory_Level": 50,
        "Stockout_Flag": 0,
        "Market_Index": 0.5129,
        "Promotion_Flag": 0,

        "Year": 2025,
        "Month": 7,
        "Quarter": 3,
        "Day_of_Week": 6,
        "Is_Weekend": 1,
        "Day_of_Month": 20,
        "Week_of_Year": 29,

        "Average_Unit_Price": 1857602.99,
        "Average_Inventory": 50.0,
        "Stockout_Rate": 0.0,
        "Promotion_Rate": 0.0,
        "Average_Market_Index": 0.5129,

        "Demand_Lag_1": 15.0,
        "Demand_Lag_2": 20.0,
        "Demand_Lag_3": 16.0,
        "Demand_Lag_7": 20.0,
        "Demand_Lag_14": 31.0,
        "Demand_Lag_28": 11.0,

        "Demand_Rolling_Mean_3": 17.0,
        "Demand_Rolling_Mean_7": 19.142857142857142,
        "Demand_Rolling_Mean_14": 20.5,
        "Demand_Rolling_Mean_28": 18.75,

        "Demand_Rolling_Std_3": 2.645751311064596,
        "Demand_Rolling_Std_7": 2.794552524023079,
        "Demand_Rolling_Std_14": 6.27265126132116,
        "Demand_Rolling_Std_28": 5.175619483408435,

        "Demand_Growth_7D": -0.2,
        "Demand_Growth_28D": 0.4545454545454545,
        "Demand_Acceleration": -0.6545454545454545,

        "Inventory_Coverage": 2.611940298507463,
        "Inventory_Pressure": 2.611940298507463,

        "Low_Inventory_Flag": 1,
        "High_Inventory_Flag": 0,
        "Stockout_Risk_Signal": 1,

        "Promotion_Lag_1": 0.0,
        "Promotion_Lag_7": 1.0,
        "Promotion_Active": 0,

        "Promotion_7D_Frequency": 2.0,

        "Demand_CV_28D": 0.2760330391151165,
        "Demand_Volatility_Flag": 0,
    }

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [sample]
    )

    # --------------------------------------------------------
    # Force exact feature order
    # --------------------------------------------------------

    X = X[
        EXPECTED_FEATURES
    ]

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    prediction = model.predict(X)

    # --------------------------------------------------------
    # Validate prediction output
    # --------------------------------------------------------

    assert prediction is not None

    assert len(prediction) == 1

    assert np.isfinite(
        prediction[0]
    ), "Prediction is not a finite number."

    assert prediction[0] >= 0, (
        f"Prediction should not be negative: "
        f"{prediction[0]}"
    )


# ============================================================
# TEST 7 — MODEL OUTPUT IS NUMERIC
# ============================================================

def test_prediction_is_numeric():

    model = joblib.load(MODEL_PATH)

    sample = {
        feature: 0
        for feature in EXPECTED_FEATURES
    }

    # Categorical values
    sample["Hospital_Type"] = "Multi_Specialty"
    sample["Hospital_Size"] = "Medium"
    sample["Territory"] = "Central"
    sample["Product_Category"] = "Radiology"

    # Reasonable numerical values
    sample["Unit_Price"] = 100000
    sample["Inventory_Level"] = 50
    sample["Market_Index"] = 0.5
    sample["Average_Unit_Price"] = 100000
    sample["Average_Inventory"] = 50
    sample["Average_Market_Index"] = 0.5

    X = pd.DataFrame(
        [sample]
    )

    X = X[
        EXPECTED_FEATURES
    ]

    prediction = model.predict(X)

    assert isinstance(
        prediction[0],
        (int, float, np.integer, np.floating)
    )