"""
Step 11.3
Production API Prediction Test

Purpose
-------
Loads a real observation from the test dataset,
constructs a valid API request, sends it to the
FastAPI prediction endpoint, and validates the response.
"""

from pathlib import Path

import json
import pandas as pd
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "test.csv"
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = (
    "http://127.0.0.1:8000/predict"
)


# ============================================================
# FEATURES EXPECTED BY API
# ============================================================

API_FEATURES = [

    "Hospital_Type",
    "Hospital_Size",
    "Territory",
    "Product_Category",

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
    
    "Unit_Price",
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
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("\n📂 Loading test dataset...")

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    print(
        f"✅ Test dataset loaded: "
        f"{len(df):,} rows"
    )

    return df


# ============================================================
# VALIDATE REQUIRED FEATURES
# ============================================================

def validate_features(df):

    print(
        "\n🔍 Validating API feature requirements..."
    )

    missing_features = [

        feature

        for feature in API_FEATURES

        if feature not in df.columns

    ]

    if missing_features:

        raise ValueError(
            "Missing API features:\n"
            + "\n".join(missing_features)
        )

    print(
        f"✅ All {len(API_FEATURES)} "
        "API features are available."
    )


# ============================================================
# CREATE REQUEST
# ============================================================

def create_request(df):

    print(
        "\n📦 Creating prediction request "
        "from a real test observation..."
    )

    row = df.iloc[0]

    request_data = {}

    for feature in API_FEATURES:

        value = row[feature]

        # Convert NumPy/Pandas values into
        # standard Python JSON-compatible values.

        if pd.isna(value):

            raise ValueError(
                f"Missing value detected in "
                f"required feature: {feature}"
            )

        if hasattr(value, "item"):

            value = value.item()

        request_data[feature] = value

    print(
        "✅ Real test observation prepared."
    )

    return request_data


# ============================================================
# SEND REQUEST
# ============================================================

def send_prediction_request(
    request_data
):

    print(
        "\n🚀 Sending request to FastAPI..."
    )

    print(
        f"Endpoint: {API_URL}"
    )

    response = requests.post(

        API_URL,

        json=request_data,

        timeout=30,

    )

    print(
        f"HTTP status: {response.status_code}"
    )

    if response.status_code != 200:

        print(
            "\n❌ API response:"
        )

        print(
            response.text
        )

        raise RuntimeError(
            "API prediction request failed."
        )

    result = response.json()

    print(
        "✅ API request successful."
    )

    return result


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result,
    actual_value,
):

    print(
        "\n" + "=" * 65
    )

    print(
        "🎯 PRODUCTION API PREDICTION"
    )

    print(
        "=" * 65
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    predicted = (

        result
        .get("prediction", {})
        .get("predicted_units_sold")

    )

    print(
        "\n📊 Test observation:"
    )

    print(
        f"   Actual Units Sold     : "
        f"{actual_value:.2f}"
    )

    print(
        f"   Predicted Units Sold  : "
        f"{predicted:.2f}"
    )

    error = abs(
        actual_value - predicted
    )

    print(
        f"   Absolute Error        : "
        f"{error:.2f}"
    )

    print(
        "\n✅ Production API prediction "
        "successfully verified."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 65
    )

    print(
        "🏥 MEDICAL DEVICE DEMAND"
    )

    print(
        "STEP 11.3 — API PREDICTION TEST"
    )

    print(
        "=" * 65
    )

    # Load test data

    df = load_test_data()

    # Validate features

    validate_features(df)

    # Prepare request

    request_data = create_request(
        df
    )

    # Actual target

    actual_value = float(
        df.iloc[0]["Units_Sold"]
    )

    # Send request

    result = send_prediction_request(
        request_data
    )

    # Display result

    display_result(
        result,
        actual_value
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "✅ STEP 11.3 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 65
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n❌ API TEST FAILED."
        )

        print(
            f"Error: {exc}"
        )