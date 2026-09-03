# 🏥 Medical Device Demand Forecasting — Production-Ready ML Pipeline

> **End-to-end machine learning and MLOps pipeline for forecasting medical-device demand using temporal validation, engineered demand signals, Gradient Boosting, model explainability, diagnostics, and a production-ready FastAPI inference service.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Pytest](https://img.shields.io/badge/tests-16%20passed-success.svg)](https://pytest.org/)
[![Model](https://img.shields.io/badge/model-Gradient%20Boosting-purple.svg)](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

---

## 📌 Project Overview

Medical-device demand is affected by factors such as historical demand, inventory levels, promotions, market conditions, hospital characteristics, and recent demand trends.

This project builds a complete machine-learning workflow to predict **medical-device units sold** while maintaining a strong focus on:

* Temporal data integrity
* Leakage prevention
* Feature engineering
* Model benchmarking
* Hyperparameter tuning
* Model explainability
* Prediction diagnostics
* Production model persistence
* REST API inference
* Automated testing

The project is designed as a **production-oriented ML engineering portfolio project**, rather than a standalone notebook experiment.

---

# 🎯 Business Problem

Medical-device organizations need better demand forecasts to help answer questions such as:

* How many units are likely to be required?
* Which demand patterns are changing?
* Is current inventory sufficient?
* Are promotions affecting demand?
* Which hospitals or product categories generate significant demand?
* How accurately can future demand be predicted?
* Can the trained model be exposed through an API for downstream applications?

### Objective

Build a machine-learning system that predicts:

> **Future medical-device demand in units sold.**

The model uses historical transactional information and engineered temporal, inventory, promotion, hospital, and demand-behavior features.

---

# 🧠 Solution Architecture

```text
                    ┌──────────────────────┐
                    │   Raw Transactional  │
                    │        Data          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Validation    │
                    │ Quality & Integrity   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Preprocessing     │
                    │ Cleaning & Formatting│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Temporal + Demand +  │
                    │ Inventory + Promotion│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │         EDA          │
                    │ Patterns & Anomalies │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Temporal Train/Test  │
                    │       Strategy       │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Model Development         │
              │                                 │
              │ Linear Regression               │
              │ Naive Baseline                 │
              │ Gradient Boosting              │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Hyperparameter       │
                    │      Tuning          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Explainability │
                    │ Feature Importance   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Diagnostics    │
                    │ Error & Residual     │
                    │ Analysis             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Production Model     │
                    │    Serialization     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │  /health + /predict  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Automated API &      │
                    │ Model Testing        │
                    └──────────────────────┘
```

---

# 📊 Dataset

The project uses a structured medical-device transactional dataset containing information across:

### Hospital attributes

* Hospital ID
* Hospital name
* Hospital type
* Hospital size
* Territory

### Product attributes

* Product ID
* Product name
* Product category
* Unit price

### Demand & revenue

* Units sold
* Total revenue
* Historical demand
* Demand growth
* Rolling demand statistics

### Inventory

* Inventory level
* Inventory coverage
* Inventory pressure
* Low-inventory flag
* High-inventory flag
* Stockout flag
* Stockout-risk signal

### Market & promotion

* Market index
* Promotion flag
* Promotion activity
* Promotion frequency
* Promotion lag features

---

# ⚙️ Feature Engineering

One of the core components of this project is the creation of **time-aware demand features**.

The engineered feature set includes:

### Lag Features

```text
Demand_Lag_1
Demand_Lag_2
Demand_Lag_3
Demand_Lag_7
Demand_Lag_14
Demand_Lag_28
```

These capture historical demand behavior at different time horizons.

### Rolling Demand Features

```text
Demand_Rolling_Mean_3
Demand_Rolling_Mean_7
Demand_Rolling_Mean_14
Demand_Rolling_Mean_28

Demand_Rolling_Std_3
Demand_Rolling_Std_7
Demand_Rolling_Std_14
Demand_Rolling_Std_28
```

These features capture recent demand level and volatility.

### Demand Dynamics

```text
Demand_Growth_7D
Demand_Growth_28D
Demand_Acceleration
Demand_CV_28D
Demand_Volatility_Flag
```

### Inventory Signals

```text
Inventory_Coverage
Inventory_Pressure
Low_Inventory_Flag
High_Inventory_Flag
Stockout_Risk_Signal
```

### Promotion Signals

```text
Promotion_Lag_1
Promotion_Lag_7
Promotion_Active
Promotion_7D_Frequency
```

### Hospital & Product Aggregations

```text
Hospital_Total_Demand
Hospital_Demand_Share
Hospital_Product_Diversity

Product_Total_Demand
Product_Demand_Share
Product_Hospital_Coverage
```

---

# 🚨 Temporal Leakage Prevention

A major focus of this project is preventing **future information from leaking into model training**.

Instead of randomly splitting observations, the project uses a temporal modeling strategy.

Conceptually:

```text
Past -----------------------------> Future

Training Data          Validation          Test
███████████████████     ████████           ███████
        │                   │                  │
        ▼                   ▼                  ▼
     Learn              Tune/Select        Final
     patterns              model          evaluation
```

This is important for demand forecasting because random train/test splitting can allow future patterns to influence training.

The pipeline explicitly validates the temporal separation between training and validation data.

---

# 🤖 Model Development

Multiple approaches were evaluated.

| Model             |        MAE |       RMSE |         R² |      MAPE |
| ----------------- | ---------: | ---------: | ---------: | --------: |
| Linear Regression |     2.3552 |     2.9755 |     0.8970 |     8.33% |
| Naive Baseline    |     8.2574 |    10.1620 |    -0.2010 |    28.10% |
| Gradient Boosting | **1.3570** | **2.0393** | **0.9573** | **4.74%** |

The Gradient Boosting model substantially outperformed both the naive baseline and linear regression.

---

# 🏆 Final Model

The selected production model is:

> **Gradient Boosting Regressor**

Final tuned configuration:

```python
{
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_samples_leaf": 3
}
```

The final model was evaluated on a held-out test set.

### Final Test Performance

```text
MAE  : 1.1825
RMSE : 1.9490
R²   : 0.9610
MAPE : 4.15%
```

### Interpretation

The final model achieved:

* **R² = 0.961** → explains approximately 96.1% of the observed variation in the held-out test target under this dataset/setup.
* **MAPE = 4.15%** → relatively low average percentage error.
* **MAE = 1.18 units** → low average absolute prediction error in the test evaluation.
* **RMSE = 1.95 units** → indicates limited larger prediction errors in the final test evaluation.

> These results describe performance on this project dataset and should not be interpreted as evidence of real-world clinical or commercial deployment performance.

---

# 🔍 Model Explainability

The project includes model explainability analysis to understand which variables contribute most strongly to predictions.

Example high-ranking features include:

```text
Demand_Rolling_Mean_3
Month
Promotion_Rate
```

Feature importance analysis helps connect model behavior with business questions such as:

* Is recent demand driving predictions?
* Are seasonal effects important?
* Do promotional patterns influence demand?
* Which operational signals deserve monitoring?

---

# 📈 Model Diagnostics

The pipeline generates multiple diagnostic outputs:

```text
actual_vs_predicted.png
residual_analysis.png
residual_distribution.png
prediction_error_analysis.csv
final_model_diagnostics.txt
final_test_predictions.csv
```

These outputs are used to evaluate:

* Prediction accuracy
* Residual behavior
* Error distribution
* Systematic prediction errors
* Actual vs predicted demand
* Model stability

---

# 🚀 Production API

The trained model is exposed through **FastAPI**.

### Health Endpoint

```http
GET /health
```

Example:

```json
{
    "status": "healthy"
}
```

### Prediction Endpoint

```http
POST /predict
```

The API accepts the model's required feature set and returns a predicted number of units.

Example response:

```json
{
    "status": "success",
    "prediction": {
        "predicted_units_sold": 42.05,
        "unit": "units"
    },
    "model": {
        "name": "Gradient Boosting Regressor",
        "version": "1.0.0",
        "features_used": 49
    }
}
```

---

# 🧪 API Validation

The production API was tested using real observations from the held-out test dataset.

The validation process checks:

* API availability
* Request schema
* Feature availability
* Prediction generation
* Prediction error
* API latency
* Multiple observations

The final API validation was successfully completed after resolving feature/schema validation issues.

---

# ✅ Automated Testing

The repository contains automated tests covering both model and API behavior.

Current test status:

```text
16 tests passed
16 tests total
0 failures
```

Testing covers areas such as:

* Model behavior
* Prediction functionality
* API health
* API prediction
* Request validation
* Production inference behavior

Run the test suite with:

```bash
python -m pytest tests -v
```

---

# 🗂️ Project Structure

```text
Medical-Devices-mlops-Pipeline/
│
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── model_preparation.py
│   ├── model_training.py
│   ├── model_tuning.py
│   ├── model_explainability.py
│   ├── model_diagnostics.py
│   ├── save_production_model.py
│   ├── evaluate.py
│   ├── train.py
│   └── api.py
│
├── tests/
│   ├── ...
│   └── ...
│
├── outputs/
│   ├── final_test_predictions.csv
│   ├── actual_vs_predicted.png
│   ├── residual_analysis.png
│   ├── residual_distribution.png
│   ├── prediction_error_analysis.csv
│   └── final_model_diagnostics.txt
│
├── requirements.txt
├── README.md
└── ...
```

---

# 💻 Local Setup

## 1. Clone the repository

```bash
git clone https://github.com/Manish04Analytics/Medical-Devices-mlops-Pipeline.git
cd Medical-Devices-mlops-Pipeline
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Run tests

```bash
python -m pytest tests -v
```

---

# 🌐 Run the API Locally

From the project root:

```bash
python -m uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Science

* Pandas
* NumPy
* SciPy
* Scikit-learn

### Machine Learning

* Linear Regression
* Gradient Boosting Regressor
* Temporal validation
* Hyperparameter tuning
* Feature importance

### Visualization

* Matplotlib
* Seaborn

### MLOps / Engineering

* Model serialization
* FastAPI
* Uvicorn
* Pytest
* Git
* GitHub

---

# 📌 Engineering Decisions

### Why Gradient Boosting?

The problem contains nonlinear relationships between demand, inventory, promotions, temporal variables, and historical demand signals.

Gradient Boosting provides a flexible nonlinear modeling approach while remaining relatively interpretable through feature importance analysis.

### Why temporal validation?

Demand forecasting is inherently time-dependent.

Random splitting can create unrealistic evaluation conditions by allowing information from later periods to influence earlier training data.

Temporal validation provides a more realistic evaluation methodology.

### Why a baseline?

A machine-learning model should demonstrate improvement over a simple reference strategy.

The naive baseline provides a benchmark against which more complex models can be evaluated.

---

# ⚠️ Limitations

This project is a portfolio/research implementation and has several limitations.

* The dataset is synthetic/simulated rather than proprietary real-world hospital data.
* Model performance may differ substantially on real-world data.
* External economic and healthcare-market variables are not comprehensively modeled.
* Forecasting performance depends on the quality and availability of historical demand signals.
* The API is currently demonstrated locally rather than as a publicly hosted production service.
* Docker/containerization and cloud deployment are not currently part of the repository's implemented workflow.

These limitations are intentionally documented to distinguish experimental results from production claims.

---

# 🔮 Future Improvements

Potential next-stage improvements include:

### MLOps

* GitHub Actions CI/CD
* Docker containerization
* Model versioning
* Experiment tracking
* Data versioning
* Model registry

### Machine Learning

* XGBoost / LightGBM benchmarking
* Time-series-specific models
* Forecast intervals
* Automated retraining
* Model drift detection
* Data drift monitoring

### Production

* Cloud deployment
* API authentication
* Request logging
* Monitoring dashboard
* Prediction monitoring
* Automated model retraining pipeline

### Business Intelligence

* Hospital-level demand dashboard
* Product-level demand forecasting
* Inventory optimization
* Stockout-risk dashboard
* Promotion impact analysis

---

# 📊 Key Takeaways

This project demonstrates an end-to-end workflow covering:

```text
Data
 ↓
Validation
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
EDA
 ↓
Temporal Validation
 ↓
Model Benchmarking
 ↓
Hyperparameter Tuning
 ↓
Explainability
 ↓
Diagnostics
 ↓
Production Model
 ↓
FastAPI
 ↓
Automated Testing
```

The final Gradient Boosting model achieved:

```text
R²   = 0.9610
MAPE = 4.15%
MAE  = 1.1825
RMSE = 1.9490
```

with **16 automated tests passing** and successful production API validation.

---

# 👨‍💻 Author

**Manish Verma**

Data Science | Machine Learning | Analytics | MLOps

GitHub:
https://github.com/Manish04Analytics

---

## ⭐ If you find this project useful

Feel free to explore the repository, review the implementation, and provide feedback.
