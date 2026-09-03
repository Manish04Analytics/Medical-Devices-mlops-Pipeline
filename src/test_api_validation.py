"""
Step 11.4
Production API Validation + Detailed Error Diagnostics

Purpose
-------
Tests the deployed FastAPI prediction API using multiple
real observations from the test dataset.

Validation includes:
- API availability
- Multiple real predictions
- Prediction validity
- Response structure
- Absolute prediction error
- API latency
- Detailed HTTP 422 diagnostics
"""

from pathlib import Path
import time

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

HEALTH_URL = (
    "http://127.0.0.1:8000/health"
)


# ============================================================
# EXACT 49 FEATURES USED BY PRODUCTION MODEL
# ============================================================

MODEL_FEATURES = [

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
# LOAD TEST DATA
# ============================================================

def load_data():

    print("\n📂 Loading test dataset...")

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    print(
        f"✅ Loaded {len(df):,} test observations."
    )

    return df


# ============================================================
# VALIDATE DATASET FEATURES
# ============================================================

def validate_dataset_features(df):

    print(
        "\n🔍 Validating dataset feature availability..."
    )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        print(
            "\n❌ Missing features:"
        )

        for feature in missing_features:

            print(
                f"   - {feature}"
            )

        raise RuntimeError(
            "Test dataset does not contain "
            "all required model features."
        )

    print(
        f"✅ All {len(MODEL_FEATURES)} "
        "model features are available."
    )


# ============================================================
# CHECK API HEALTH
# ============================================================

def check_api_health():

    print(
        "\n🏥 Checking API health..."
    )

    try:

        response = requests.get(
            HEALTH_URL,
            timeout=10
        )

    except requests.exceptions.RequestException as exc:

        raise RuntimeError(
            "Could not connect to FastAPI.\n"
            f"Error: {exc}"
        )

    if response.status_code != 200:

        raise RuntimeError(
            "API health check failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text}"
        )

    result = response.json()

    if result.get("status") != "healthy":

        raise RuntimeError(
            f"API reports unhealthy status:\n"
            f"{result}"
        )

    print(
        "✅ API health check passed."
    )


# ============================================================
# BUILD REQUEST FROM REAL DATASET ROW
# ============================================================

def build_request(row):

    request_data = {}

    for feature in MODEL_FEATURES:

        value = row[feature]

        # Check missing values

        if pd.isna(value):

            raise ValueError(
                f"Missing value detected in "
                f"feature: {feature}"
            )

        # Convert NumPy/Pandas scalar
        # into normal Python value

        if hasattr(value, "item"):

            value = value.item()

        request_data[feature] = value

    return request_data


# ============================================================
# PRINT 422 DIAGNOSTIC INFORMATION
# ============================================================

def print_validation_error(
    row_index,
    response,
    request_data
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"❌ Row {row_index}: "
        f"HTTP {response.status_code}"
    )

    print(
        "🔎 FastAPI validation details:"
    )

    print(
        response.text
    )

    # --------------------------------------------------------
    # Try to parse FastAPI JSON response
    # --------------------------------------------------------

    try:

        error_json = response.json()

        details = error_json.get(
            "detail"
        )

        if isinstance(details, list):

            print(
                "\n📋 Parsed validation errors:"
            )

            for error in details:

                location = error.get(
                    "loc",
                    []
                )

                message = error.get(
                    "msg",
                    "Unknown validation error"
                )

                error_type = error.get(
                    "type",
                    "unknown"
                )

                print(
                    f"   Field: {location}"
                )

                print(
                    f"   Message: {message}"
                )

                print(
                    f"   Type: {error_type}"
                )

                print()

    except Exception:

        print(
            "⚠️ Could not parse validation "
            "response as JSON."
        )

    # --------------------------------------------------------
    # Display values sent to API
    # --------------------------------------------------------

    print(
        "📦 Values sent for this row:"
    )

    for feature, value in request_data.items():

        print(
            f"   {feature}: {value}"
        )

    print(
        "-" * 70
    )


# ============================================================
# TEST MULTIPLE PREDICTIONS
# ============================================================

def test_predictions(df):

    print(
        "\n🚀 Testing multiple real observations..."
    )

    # Test first 20 rows

    sample_size = min(
        20,
        len(df)
    )

    sample = df.head(
        sample_size
    )

    results = []

    successful = 0

    failed = 0

    total_latency = 0.0

    # --------------------------------------------------------
    # Process each observation
    # --------------------------------------------------------

    for index, row in sample.iterrows():

        try:

            request_data = build_request(
                row
            )

        except Exception as exc:

            failed += 1

            print(
                f"\n❌ Row {index}: "
                f"Could not build request."
            )

            print(
                f"Error: {exc}"
            )

            continue

        # ----------------------------------------------------
        # Send API request
        # ----------------------------------------------------

        start_time = time.perf_counter()

        try:

            response = requests.post(

                API_URL,

                json=request_data,

                timeout=30

            )

        except requests.exceptions.RequestException as exc:

            failed += 1

            print(
                f"\n❌ Row {index}: "
                "API connection failed."
            )

            print(
                f"Error: {exc}"
            )

            continue

        end_time = time.perf_counter()

        latency = (
            end_time - start_time
        )

        total_latency += latency

        # ----------------------------------------------------
        # Handle HTTP errors
        # ----------------------------------------------------

        if response.status_code != 200:

            failed += 1

            print_validation_error(

                index,

                response,

                request_data

            )

            continue

        # ----------------------------------------------------
        # Parse response
        # ----------------------------------------------------

        try:

            result = response.json()

        except Exception as exc:

            failed += 1

            print(
                f"\n❌ Row {index}: "
                "Invalid JSON response."
            )

            print(
                f"Error: {exc}"
            )

            continue

        # ----------------------------------------------------
        # Extract prediction
        # ----------------------------------------------------

        prediction_section = (
            result.get(
                "prediction",
                {}
            )
        )

        predicted = (
            prediction_section.get(
                "predicted_units_sold"
            )
        )

        # ----------------------------------------------------
        # Validate prediction exists
        # ----------------------------------------------------

        if predicted is None:

            failed += 1

            print(
                f"\n❌ Row {index}: "
                "Prediction missing from response."
            )

            print(
                response.text
            )

            continue

        # ----------------------------------------------------
        # Convert prediction
        # ----------------------------------------------------

        try:

            predicted = float(
                predicted
            )

        except Exception:

            failed += 1

            print(
                f"\n❌ Row {index}: "
                "Prediction is not numeric."
            )

            continue

        # ----------------------------------------------------
        # Validate non-negative prediction
        # ----------------------------------------------------

        if predicted < 0:

            failed += 1

            print(
                f"\n❌ Row {index}: "
                "Negative prediction returned."
            )

            continue

        # ----------------------------------------------------
        # Actual value
        # ----------------------------------------------------

        actual = float(
            row["Units_Sold"]
        )

        # ----------------------------------------------------
        # Absolute error
        # ----------------------------------------------------

        error = abs(
            actual - predicted
        )

        # ----------------------------------------------------
        # Store successful result
        # ----------------------------------------------------

        successful += 1

        results.append({

            "row_index":
                int(index),

            "actual_units":
                actual,

            "predicted_units":
                predicted,

            "absolute_error":
                error,

            "latency_seconds":
                latency,

        })

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print(
            f"✅ Row {index:4d} | "
            f"Actual: {actual:7.2f} | "
            f"Predicted: {predicted:7.2f} | "
            f"Error: {error:7.2f} | "
            f"Latency: {latency:.4f}s"
        )

    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 API VALIDATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total observations tested : "
        f"{sample_size}"
    )

    print(
        f"Successful predictions     : "
        f"{successful}"
    )

    print(
        f"Failed predictions         : "
        f"{failed}"
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    if successful > 0:

        successful_df = pd.DataFrame(
            results
        )

        mae = (
            successful_df[
                "absolute_error"
            ].mean()
        )

        avg_latency = (
            successful_df[
                "latency_seconds"
            ].mean()
        )

        max_latency = (
            successful_df[
                "latency_seconds"
            ].max()
        )

        print(
            f"Average absolute error     : "
            f"{mae:.4f}"
        )

        print(
            f"Average API latency        : "
            f"{avg_latency:.4f} sec"
        )

        print(
            f"Maximum API latency        : "
            f"{max_latency:.4f} sec"
        )

    else:

        print(
            "\n⚠️ No successful predictions "
            "were generated."
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    if failed == 0:

        print(
            "🎉 ALL API PREDICTIONS PASSED."
        )

        print(
            "✅ Production API validation successful."
        )

    else:

        print(
            "⚠️ API VALIDATION REQUIRES FIXES."
        )

        print(
            f"Successful: {successful}"
        )

        print(
            f"Failed: {failed}"
        )

        print(
            "The detailed HTTP errors above "
            "identify the exact validation issue."
        )

    print(
        "=" * 70
    )

    return failed


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "🏥 MEDICAL DEVICE DEMAND"
    )

    print(
        "STEP 11.4 — PRODUCTION API VALIDATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Validate required features
    # --------------------------------------------------------

    validate_dataset_features(
        df
    )

    # --------------------------------------------------------
    # Check API
    # --------------------------------------------------------

    check_api_health()

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    failed = test_predictions(
        df
    )

    # --------------------------------------------------------
    # Final project status
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    if failed == 0:

        print(
            "✅ STEP 11.4 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "⚠️ STEP 11.4 DIAGNOSTICS COMPLETED"
        )

        print(
            "Review the detailed validation "
            "errors above before fixing the API."
        )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "\n❌ API VALIDATION SCRIPT FAILED."
        )

        print(
            f"Error: {exc}"
        )