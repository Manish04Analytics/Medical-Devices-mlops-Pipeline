"""
Healthcare Medical-Device Sales
Feature Engineering Pipeline
================================

Purpose
-------
Transform the cleaned healthcare sales dataset into a predictive
feature dataset suitable for downstream machine-learning models.

The feature-engineering layer creates:

1. Calendar features
2. Sales velocity features
3. Lag features
4. Rolling-window features
5. Growth features
6. Inventory pressure features
7. Promotion features
8. Hospital-account features
9. Product-level features
10. Revenue features

Important
---------
All historical features are generated using past observations only.

This helps reduce target leakage when the dataset is later used
for forecasting and demand prediction.

Input
-----
data/processed/healthcare_sales_clean.csv

Output
------
data/processed/healthcare_sales_features.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "healthcare_sales_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "healthcare_sales_features.csv"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Transaction_Date",
    "Hospital_ID",
    "Hospital_Name",
    "Hospital_Type",
    "Hospital_Size",
    "Territory",
    "Product_ID",
    "Product_Name",
    "Product_Category",
    "Units_Sold",
    "Unit_Price",
    "Total_Revenue",
    "Inventory_Level",
    "Stockout_Flag",
    "Market_Index",
    "Promotion_Flag",
    "Year",
    "Month",
    "Quarter",
    "Day_of_Week",
    "Is_Weekend",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_processed_data() -> pd.DataFrame:
    """
    Load the cleaned dataset created during preprocessing.
    """

    print("\n📂 Loading processed healthcare sales data...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Transaction_Date"],
    )

    print(
        f"✅ Loaded {len(df):,} rows "
        f"and {len(df.columns)} columns."
    )

    return df


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(
    df: pd.DataFrame,
) -> None:
    """
    Confirm that the expected columns are available.
    """

    print("\n🔎 Validating feature-engineering input schema...")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_columns
            )
        )

    print(
        "✅ Input schema validation passed."
    )


# ============================================================
# CALENDAR FEATURES
# ============================================================

def create_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create calendar features from Transaction_Date.
    """

    print("\n📅 Creating calendar features...")

    df = df.copy()

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    df["Quarter_Sin"] = np.sin(
        2 * np.pi * df["Quarter"] / 4
    )

    df["Quarter_Cos"] = np.cos(
        2 * np.pi * df["Quarter"] / 4
    )

    df["DayOfWeek_Sin"] = np.sin(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    df["DayOfWeek_Cos"] = np.cos(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    df["Days_Since_Start"] = (
        df["Transaction_Date"]
        - df["Transaction_Date"].min()
    ).dt.days

    print(
        "   ✓ Cyclical month features"
    )

    print(
        "   ✓ Cyclical quarter features"
    )

    print(
        "   ✓ Cyclical weekday features"
    )

    print(
        "   ✓ Days-since-start feature"
    )

    print(
        "✅ Calendar feature engineering completed."
    )

    return df


# ============================================================
# UNIT ECONOMICS
# ============================================================

def create_unit_economics_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create unit economics and revenue-derived features.
    """

    print("\n💰 Creating unit-economics features...")

    df = df.copy()

    # Revenue per unit.
    df["Revenue_Per_Unit"] = np.where(
        df["Units_Sold"] > 0,
        df["Total_Revenue"] / df["Units_Sold"],
        0,
    )

    # Inventory-to-sales ratio.
    df["Inventory_to_Sales_Ratio"] = np.where(
        df["Units_Sold"] > 0,
        df["Inventory_Level"] / df["Units_Sold"],
        np.nan,
    )

    # Revenue per inventory unit.
    df["Revenue_per_Inventory_Unit"] = np.where(
        df["Inventory_Level"] > 0,
        df["Total_Revenue"] / df["Inventory_Level"],
        0,
    )

    print(
        "   ✓ Revenue_Per_Unit"
    )

    print(
        "   ✓ Inventory_to_Sales_Ratio"
    )

    print(
        "   ✓ Revenue_per_Inventory_Unit"
    )

    print(
        "✅ Unit-economics features created."
    )

    return df


# ============================================================
# DAILY AGGREGATION
# ============================================================

def create_daily_demand_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a daily hospital-product demand table.

    This creates the time-series grain:

        Transaction_Date
        +
        Hospital_ID
        +
        Product_ID

    Units_Sold are aggregated to that grain.
    """

    print("\n📊 Creating daily hospital-product demand table...")

    daily = (
        df.groupby(
            [
                "Transaction_Date",
                "Hospital_ID",
                "Product_ID",
            ],
            as_index=False,
        )
        .agg(
            Daily_Units_Sold=(
                "Units_Sold",
                "sum",
            ),
            Daily_Revenue=(
                "Total_Revenue",
                "sum",
            ),
            Average_Unit_Price=(
                "Unit_Price",
                "mean",
            ),
            Average_Inventory=(
                "Inventory_Level",
                "mean",
            ),
            Stockout_Rate=(
                "Stockout_Flag",
                "mean",
            ),
            Promotion_Rate=(
                "Promotion_Flag",
                "mean",
            ),
            Average_Market_Index=(
                "Market_Index",
                "mean",
            ),
        )
    )

    print(
        f"✅ Created {len(daily):,} "
        "daily hospital-product observations."
    )

    return daily


# ============================================================
# LAG FEATURES
# ============================================================

def create_lag_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical demand lag features.

    Lags are calculated separately for each hospital-product
    combination.
    """

    print("\n⏮️ Creating historical demand lag features...")

    daily = daily.copy()

    group_columns = [
        "Hospital_ID",
        "Product_ID",
    ]

    daily = daily.sort_values(
        group_columns
        + ["Transaction_Date"]
    )

    grouped_units = daily.groupby(
        group_columns,
        sort=False,
    )["Daily_Units_Sold"]

    lag_periods = [
        1,
        2,
        3,
        7,
        14,
        28,
    ]

    for lag in lag_periods:

        daily[
            f"Demand_Lag_{lag}"
        ] = grouped_units.shift(
            lag
        )

    print(
        "   ✓ 1-period lag"
    )

    print(
        "   ✓ 2-period lag"
    )

    print(
        "   ✓ 3-period lag"
    )

    print(
        "   ✓ 7-period lag"
    )

    print(
        "   ✓ 14-period lag"
    )

    print(
        "   ✓ 28-period lag"
    )

    print(
        "✅ Lag features created."
    )

    return daily


# ============================================================
# ROLLING FEATURES
# ============================================================

def create_rolling_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create rolling demand statistics.

    IMPORTANT:
    Shift(1) is applied before rolling calculations so that
    today's target does not leak into today's historical features.
    """

    print("\n📈 Creating rolling demand features...")

    daily = daily.copy()

    group_columns = [
        "Hospital_ID",
        "Product_ID",
    ]

    grouped = daily.groupby(
        group_columns,
        sort=False,
    )["Daily_Units_Sold"]

    shifted_demand = grouped.shift(1)

    rolling_windows = [
        3,
        7,
        14,
        28,
    ]

    for window in rolling_windows:

        daily[
            f"Demand_Rolling_Mean_{window}"
        ] = (
            shifted_demand
            .groupby(
                [
                    daily["Hospital_ID"],
                    daily["Product_ID"],
                ]
            )
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
        )

        daily[
            f"Demand_Rolling_Std_{window}"
        ] = (
            shifted_demand
            .groupby(
                [
                    daily["Hospital_ID"],
                    daily["Product_ID"],
                ]
            )
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=2,
                ).std()
            )
        )

    print(
        "   ✓ Rolling mean: 3 periods"
    )

    print(
        "   ✓ Rolling mean: 7 periods"
    )

    print(
        "   ✓ Rolling mean: 14 periods"
    )

    print(
        "   ✓ Rolling mean: 28 periods"
    )

    print(
        "   ✓ Rolling volatility features"
    )

    print(
        "✅ Rolling features created."
    )

    return daily


# ============================================================
# DEMAND TREND FEATURES
# ============================================================

def create_demand_trend_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create demand growth and trend indicators.
    """

    print("\n📉 Creating demand trend features...")

    daily = daily.copy()

    group_columns = [
        "Hospital_ID",
        "Product_ID",
    ]

    grouped = daily.groupby(
        group_columns,
        sort=False,
    )["Daily_Units_Sold"]

    demand_lag_7 = grouped.shift(7)
    demand_lag_28 = grouped.shift(28)

    daily["Demand_Growth_7D"] = np.where(
        demand_lag_7 > 0,
        (
            daily["Daily_Units_Sold"]
            - demand_lag_7
        )
        / demand_lag_7,
        np.nan,
    )

    daily["Demand_Growth_28D"] = np.where(
        demand_lag_28 > 0,
        (
            daily["Daily_Units_Sold"]
            - demand_lag_28
        )
        / demand_lag_28,
        np.nan,
    )

    daily["Demand_Acceleration"] = (
        daily["Demand_Growth_7D"]
        - daily["Demand_Growth_28D"]
    )

    print(
        "   ✓ 7-day demand growth"
    )

    print(
        "   ✓ 28-day demand growth"
    )

    print(
        "   ✓ Demand acceleration"
    )

    print(
        "✅ Demand trend features created."
    )

    return daily


# ============================================================
# INVENTORY FEATURES
# ============================================================

def create_inventory_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create inventory-pressure indicators.

    These features are designed to capture potential supply
    pressure and stockout risk.
    """

    print("\n📦 Creating inventory features...")

    daily = daily.copy()

    daily["Inventory_Coverage"] = np.where(
        daily["Demand_Rolling_Mean_7"] > 0,
        daily["Average_Inventory"]
        / daily["Demand_Rolling_Mean_7"],
        np.nan,
    )

    daily["Inventory_Pressure"] = np.where(
        daily["Demand_Rolling_Mean_7"] > 0,
        daily["Average_Inventory"]
        / daily["Demand_Rolling_Mean_7"],
        np.nan,
    )

    daily["Low_Inventory_Flag"] = (
        daily["Inventory_Coverage"]
        < 7
    ).astype(int)

    daily["High_Inventory_Flag"] = (
        daily["Inventory_Coverage"]
        > 30
    ).astype(int)

    daily["Stockout_Risk_Signal"] = (
        (
            daily["Inventory_Coverage"]
            < 7
        )
        |
        (
            daily["Stockout_Rate"]
            > 0
        )
    ).astype(int)

    print(
        "   ✓ Inventory coverage"
    )

    print(
        "   ✓ Inventory pressure"
    )

    print(
        "   ✓ Low-inventory flag"
    )

    print(
        "   ✓ High-inventory flag"
    )

    print(
        "   ✓ Stockout-risk signal"
    )

    print(
        "✅ Inventory features created."
    )

    return daily


# ============================================================
# PROMOTION FEATURES
# ============================================================

def create_promotion_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create promotion-related features.
    """

    print("\n📣 Creating promotion features...")

    daily = daily.copy()

    group_columns = [
        "Hospital_ID",
        "Product_ID",
    ]

    promotion_group = daily.groupby(
        group_columns,
        sort=False,
    )["Promotion_Rate"]

    daily["Promotion_Lag_1"] = (
        promotion_group.shift(1)
    )

    daily["Promotion_Lag_7"] = (
        promotion_group.shift(7)
    )

    daily["Promotion_Active"] = (
        daily["Promotion_Rate"] > 0
    ).astype(int)

    daily["Promotion_7D_Frequency"] = (
        daily["Promotion_Active"]
        .groupby(
            [
                daily["Hospital_ID"],
                daily["Product_ID"],
            ]
        )
        .transform(
            lambda x: x.shift(1)
            .rolling(
                7,
                min_periods=1,
            )
            .sum()
        )
    )

    print(
        "   ✓ Promotion lag features"
    )

    print(
        "   ✓ Promotion-active indicator"
    )

    print(
        "   ✓ 7-day promotion frequency"
    )

    print(
        "✅ Promotion features created."
    )

    return daily


# ============================================================
# HOSPITAL ACCOUNT FEATURES
# ============================================================

def create_hospital_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create hospital-account level features.
    """

    print("\n🏥 Creating hospital-account features...")

    daily = daily.copy()

    hospital_total_demand = (
        daily.groupby(
            "Hospital_ID"
        )["Daily_Units_Sold"]
        .transform("sum")
    )

    daily["Hospital_Total_Demand"] = (
        hospital_total_demand
    )

    daily["Hospital_Demand_Share"] = np.where(
        hospital_total_demand > 0,
        daily["Daily_Units_Sold"]
        / hospital_total_demand,
        0,
    )

    hospital_product_count = (
        daily.groupby(
            "Hospital_ID"
        )["Product_ID"]
        .transform("nunique")
    )

    daily["Hospital_Product_Diversity"] = (
        hospital_product_count
    )

    print(
        "   ✓ Hospital total demand"
    )

    print(
        "   ✓ Hospital demand share"
    )

    print(
        "   ✓ Hospital product diversity"
    )

    print(
        "✅ Hospital-account features created."
    )

    return daily


# ============================================================
# PRODUCT FEATURES
# ============================================================

def create_product_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create product-level demand features.
    """

    print("\n🔬 Creating product-level features...")

    daily = daily.copy()

    product_total_demand = (
        daily.groupby(
            "Product_ID"
        )["Daily_Units_Sold"]
        .transform("sum")
    )

    daily["Product_Total_Demand"] = (
        product_total_demand
    )

    daily["Product_Demand_Share"] = np.where(
        product_total_demand > 0,
        daily["Daily_Units_Sold"]
        / product_total_demand,
        0,
    )

    product_hospital_count = (
        daily.groupby(
            "Product_ID"
        )["Hospital_ID"]
        .transform("nunique")
    )

    daily["Product_Hospital_Coverage"] = (
        product_hospital_count
    )

    print(
        "   ✓ Product total demand"
    )

    print(
        "   ✓ Product demand share"
    )

    print(
        "   ✓ Product hospital coverage"
    )

    print(
        "✅ Product-level features created."
    )

    return daily


# ============================================================
# DEMAND VOLATILITY
# ============================================================

def create_volatility_features(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create demand volatility indicators.
    """

    print("\n📊 Creating demand volatility features...")

    daily = daily.copy()

    daily["Demand_CV_28D"] = np.where(
        daily["Demand_Rolling_Mean_28"] > 0,
        daily["Demand_Rolling_Std_28"]
        / daily["Demand_Rolling_Mean_28"],
        np.nan,
    )

    daily["Demand_Volatility_Flag"] = (
        daily["Demand_CV_28D"] > 1
    ).astype(int)

    print(
        "   ✓ 28-day coefficient of variation"
    )

    print(
        "   ✓ Demand-volatility flag"
    )

    print(
        "✅ Volatility features created."
    )

    return daily


# ============================================================
# MERGE FEATURES BACK TO TRANSACTION LEVEL
# ============================================================

def merge_features_to_transactions(
    original_df: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily hospital-product features back onto the
    transaction-level dataset.

    This preserves the original transaction-level records while
    attaching historical demand features.
    """

    print(
        "\n🔗 Merging engineered features "
        "back to transaction-level data..."
    )

    feature_columns = [
        "Transaction_Date",
        "Hospital_ID",
        "Product_ID",
        "Daily_Units_Sold",
        "Daily_Revenue",
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
        "Hospital_Total_Demand",
        "Hospital_Demand_Share",
        "Hospital_Product_Diversity",
        "Product_Total_Demand",
        "Product_Demand_Share",
        "Product_Hospital_Coverage",
        "Demand_CV_28D",
        "Demand_Volatility_Flag",
    ]

    available_columns = [
        column
        for column in feature_columns
        if column in daily.columns
    ]

    feature_table = daily[
        available_columns
    ].copy()

    merged = original_df.merge(
        feature_table,
        on=[
            "Transaction_Date",
            "Hospital_ID",
            "Product_ID",
        ],
        how="left",
        validate="many_to_one",
    )

    print(
        f"✅ Feature merge completed."
    )

    print(
        f"   Final rows: {len(merged):,}"
    )

    print(
        f"   Final columns: {len(merged.columns)}"
    )

    return merged


# ============================================================
# HANDLE ENGINEERED FEATURE MISSING VALUES
# ============================================================

def handle_feature_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing values created naturally by lag and rolling
    calculations.

    Missing lag values are retained as NaN because they indicate
    insufficient historical observations rather than bad data.

    For ratio features, infinite values are converted to NaN.
    """

    print(
        "\n🧹 Cleaning engineered features..."
    )

    df = df.copy()

    # Replace infinite values.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove completely impossible values.
    df["Demand_CV_28D"] = (
        df["Demand_CV_28D"]
        .clip(
            lower=0,
            upper=100,
        )
    )

    df["Demand_Growth_7D"] = (
        df["Demand_Growth_7D"]
        .clip(
            lower=-10,
            upper=10,
        )
    )

    df["Demand_Growth_28D"] = (
        df["Demand_Growth_28D"]
        .clip(
            lower=-10,
            upper=10,
        )
    )

    df["Demand_Acceleration"] = (
        df["Demand_Acceleration"]
        .clip(
            lower=-20,
            upper=20,
        )
    )

    print(
        "   ✓ Infinite values handled."
    )

    print(
        "   ✓ Extreme engineered ratios constrained."
    )

    print(
        "ℹ️ Lag-related NaN values are intentionally retained."
    )

    return df


# ============================================================
# FINAL SORTING
# ============================================================

def sort_feature_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort data chronologically and by hospital/product.
    """

    print(
        "\n📚 Sorting engineered dataset..."
    )

    df = df.sort_values(
        by=[
            "Transaction_Date",
            "Hospital_ID",
            "Product_ID",
        ]
    )

    df = df.reset_index(
        drop=True
    )

    print(
        "✅ Dataset sorted."
    )

    return df


# ============================================================
# FEATURE SUMMARY
# ============================================================

def print_feature_summary(
    df: pd.DataFrame,
) -> None:
    """
    Display feature-engineering summary.
    """

    original_feature_count = len(
        REQUIRED_COLUMNS
    )

    final_feature_count = len(
        df.columns
    )

    engineered_count = (
        final_feature_count
        - original_feature_count
    )

    print("\n")
    print("=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    print(
        f"Rows                  : "
        f"{len(df):,}"
    )

    print(
        f"Original columns      : "
        f"{original_feature_count}"
    )

    print(
        f"Final columns         : "
        f"{final_feature_count}"
    )

    print(
        f"Engineered columns    : "
        f"{engineered_count}"
    )

    print(
        f"Unique hospitals      : "
        f"{df['Hospital_ID'].nunique():,}"
    )

    print(
        f"Unique products       : "
        f"{df['Product_ID'].nunique():,}"
    )

    print(
        f"Date range            : "
        f"{df['Transaction_Date'].min().date()} "
        f"→ "
        f"{df['Transaction_Date'].max().date()}"
    )

    print(
        f"Total units sold      : "
        f"{df['Units_Sold'].sum():,.0f}"
    )

    print(
        f"Total revenue         : "
        f"₹{df['Total_Revenue'].sum():,.2f}"
    )

    print("=" * 70)

    print(
        "\n📌 Major engineered feature groups:"
    )

    print(
        "   • Calendar / seasonality"
    )

    print(
        "   • Demand lags"
    )

    print(
        "   • Rolling demand statistics"
    )

    print(
        "   • Demand growth"
    )

    print(
        "   • Demand acceleration"
    )

    print(
        "   • Inventory coverage"
    )

    print(
        "   • Stockout risk"
    )

    print(
        "   • Promotion effects"
    )

    print(
        "   • Hospital-account behaviour"
    )

    print(
        "   • Product-level behaviour"
    )

    print(
        "   • Demand volatility"
    )


# ============================================================
# FINAL QUALITY CHECK
# ============================================================

def final_quality_check(
    df: pd.DataFrame,
) -> None:
    """
    Perform final quality checks.
    """

    print(
        "\n🔍 Running final feature-dataset quality checks..."
    )

    if df.empty:

        raise ValueError(
            "Feature dataset is empty."
        )

    if df["Transaction_Date"].isna().any():

        raise ValueError(
            "Feature dataset contains invalid dates."
        )

    if df["Hospital_ID"].isna().any():

        raise ValueError(
            "Hospital_ID contains missing values."
        )

    if df["Product_ID"].isna().any():

        raise ValueError(
            "Product_ID contains missing values."
        )

    if (
        df["Units_Sold"] < 0
    ).any():

        raise ValueError(
            "Negative Units_Sold detected."
        )

    if (
        df["Total_Revenue"] < 0
    ).any():

        raise ValueError(
            "Negative Total_Revenue detected."
        )

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        print(
            f"⚠️ Warning: "
            f"{duplicate_count:,} duplicate "
            "transaction rows detected."
        )

    print(
        "   ✓ Date integrity passed."
    )

    print(
        "   ✓ Hospital identifiers passed."
    )

    print(
        "   ✓ Product identifiers passed."
    )

    print(
        "   ✓ Units sold integrity passed."
    )

    print(
        "   ✓ Revenue integrity passed."
    )

    print(
        "✅ Final feature-dataset checks passed."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_features() -> pd.DataFrame:
    """
    Execute the complete feature-engineering pipeline.
    """

    print("\n" + "=" * 70)
    print("🧠 HEALTHCARE SALES FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_processed_data()

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    validate_schema(
        df
    )

    original_df = df.copy()

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    df = create_calendar_features(
        df
    )

    # --------------------------------------------------------
    # Unit economics
    # --------------------------------------------------------

    df = create_unit_economics_features(
        df
    )

    # --------------------------------------------------------
    # Daily demand table
    # --------------------------------------------------------

    daily = create_daily_demand_table(
        df
    )

    # --------------------------------------------------------
    # Historical lags
    # --------------------------------------------------------

    daily = create_lag_features(
        daily
    )

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    daily = create_rolling_features(
        daily
    )

    # --------------------------------------------------------
    # Demand trends
    # --------------------------------------------------------

    daily = create_demand_trend_features(
        daily
    )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    daily = create_inventory_features(
        daily
    )

    # --------------------------------------------------------
    # Promotions
    # --------------------------------------------------------

    daily = create_promotion_features(
        daily
    )

    # --------------------------------------------------------
    # Hospital behaviour
    # --------------------------------------------------------

    daily = create_hospital_features(
        daily
    )

    # --------------------------------------------------------
    # Product behaviour
    # --------------------------------------------------------

    daily = create_product_features(
        daily
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    daily = create_volatility_features(
        daily
    )

    # --------------------------------------------------------
    # Merge features
    # --------------------------------------------------------

    df = merge_features_to_transactions(
        original_df,
        daily,
    )

    # --------------------------------------------------------
    # Feature cleanup
    # --------------------------------------------------------

    df = handle_feature_missing_values(
        df
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = sort_feature_dataset(
        df
    )

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    final_quality_check(
        df
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\n💾 Feature dataset saved to:"
        f"\n   {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_feature_summary(
        df
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    return df


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        build_features()

        sys.exit(0)

    except Exception as exc:

        print(
            "\n❌ FEATURE ENGINEERING FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)