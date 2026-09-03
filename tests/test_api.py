"""
Automated API Tests
===================

Step 12.3

Tests the production FastAPI service:

1. Health endpoint
2. Model information endpoint
3. Prediction endpoint
4. Prediction response structure
5. Prediction value validity
6. Invalid request handling
"""

import requests
import math


BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# REALISTIC 49-FEATURE PRODUCTION REQUEST
# ============================================================

VALID_PAYLOAD = {
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

    "Promotion_7D_Frequency": 1.0,

    "Demand_CV_28D": 0.2760330391151165,
    "Demand_Volatility_Flag": 0,
}


# ============================================================
# TEST 1 — HEALTH CHECK
# ============================================================

def test_health_endpoint():

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)

    print("\nHealth response:")
    print(data)


# ============================================================
# TEST 2 — OPENAPI IS AVAILABLE
# ============================================================

def test_openapi_endpoint():

    response = requests.get(
        f"{BASE_URL}/openapi.json",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert "openapi" in data
    assert "paths" in data

    print("\nOpenAPI schema loaded successfully.")


# ============================================================
# TEST 3 — PREDICTION ENDPOINT EXISTS
# ============================================================

def test_prediction_endpoint_exists():

    response = requests.post(
        f"{BASE_URL}/predict",
        json=VALID_PAYLOAD,
        timeout=10
    )

    assert response.status_code == 200

    print("\nPrediction endpoint returned HTTP 200.")


# ============================================================
# TEST 4 — PREDICTION RESPONSE STRUCTURE
# ============================================================

def test_prediction_response_structure():

    response = requests.post(
        f"{BASE_URL}/predict",
        json=VALID_PAYLOAD,
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    # Top-level response
    assert "status" in data
    assert "prediction" in data
    assert "model" in data

    # Prediction section
    assert "predicted_units_sold" in data["prediction"]
    assert "unit" in data["prediction"]

    # Model section
    assert "name" in data["model"]
    assert "version" in data["model"]
    assert "features_used" in data["model"]

    print("\nPrediction response:")
    print(data)


# ============================================================
# TEST 5 — PREDICTION VALUE IS VALID
# ============================================================

def test_prediction_value():

    response = requests.post(
        f"{BASE_URL}/predict",
        json=VALID_PAYLOAD,
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    prediction = data["prediction"]["predicted_units_sold"]

    # Must be numeric
    assert isinstance(
        prediction,
        (int, float)
    )

    # Must not be NaN or infinity
    assert math.isfinite(prediction)

    # Demand cannot be negative
    assert prediction >= 0

    print(
        f"\nPredicted units sold: {prediction}"
    )


# ============================================================
# TEST 6 — MODEL INFORMATION IS CORRECT
# ============================================================

def test_model_information():

    response = requests.post(
        f"{BASE_URL}/predict",
        json=VALID_PAYLOAD,
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    model_info = data["model"]

    assert model_info["name"] == (
        "Gradient Boosting Regressor"
    )

    assert model_info["version"] == "1.0.0"

    assert model_info["features_used"] == 49

    print("\nModel information validated successfully.")


# ============================================================
# TEST 7 — INVALID REQUEST IS REJECTED
# ============================================================

def test_invalid_request_rejected():

    invalid_payload = VALID_PAYLOAD.copy()

    # Remove a required feature
    del invalid_payload["Unit_Price"]

    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_payload,
        timeout=10
    )

    # FastAPI should reject invalid input
    assert response.status_code in [400, 422]

    print(
        f"\nInvalid request correctly rejected "
        f"with HTTP {response.status_code}."
    )


# ============================================================
# TEST 8 — WRONG FEATURE TYPE IS REJECTED
# ============================================================

def test_invalid_feature_type_rejected():

    invalid_payload = VALID_PAYLOAD.copy()

    # Intentionally send an invalid type
    invalid_payload["Inventory_Level"] = "INVALID"

    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_payload,
        timeout=10
    )

    assert response.status_code in [400, 422]

    print(
        f"\nInvalid feature type correctly rejected "
        f"with HTTP {response.status_code}."
    )


# ============================================================
# TEST 9 — PREDICTION IS REPEATABLE
# ============================================================

def test_prediction_repeatability():

    response_1 = requests.post(
        f"{BASE_URL}/predict",
        json=VALID_PAYLOAD,
        timeout=10
    )

    response_2 = requests.post(
        f"{BASE_URL}/predict",
        json=VALID_PAYLOAD,
        timeout=10
    )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    prediction_1 = response_1.json()[
        "prediction"
    ]["predicted_units_sold"]

    prediction_2 = response_2.json()[
        "prediction"
    ]["predicted_units_sold"]

    assert prediction_1 == prediction_2

    print(
        f"\nRepeatability test passed: "
        f"{prediction_1}"
    )