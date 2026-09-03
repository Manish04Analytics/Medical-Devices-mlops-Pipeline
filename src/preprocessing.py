"""
Healthcare Sales Data Preprocessing
===================================

Purpose
-------
Transform the validated raw healthcare sales dataset into a clean,
consistent and machine-learning-ready dataset.

Processing steps
----------------
1. Load raw data
2. Standardize column names
3. Convert dates
4. Handle missing numerical values
5. Handle missing categorical values
6. Remove duplicate records
7. Apply business-rule corrections
8. Create derived time variables
9. Validate the processed dataset
10. Save the clean dataset

Input
-----
data/raw/healthcare_sales.csv

Output
------
data/processed/healthcare_sales_clean.csv

IMPORTANT
---------
The source dataset is synthetic and created only for portfolio,
data-science and MLOps experimentation.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "healthcare_sales.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "healthcare_sales_clean.csv"
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
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
# NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

NUMERICAL_COLUMNS = [
    "Units_Sold",
    "Unit_Price",
    "Total_Revenue",
    "Inventory_Level",
    "Market_Index",
]


CATEGORICAL_COLUMNS = [
    "Hospital_ID",
    "Hospital_Name",
    "Hospital_Type",
    "Hospital_Size",
    "Territory",
    "Product_ID",
    "Product_Name",
    "Product_Category",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw healthcare sales dataset.
    """

    print("\n📂 Loading raw dataset...")

    if not RAW_DATA_FILE.exists():

        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_FILE}"
        )

    df = pd.read_csv(
        RAW_DATA_FILE
    )

    print(
        f"✅ Loaded {len(df):,} rows "
        f"and {len(df.columns)} columns."
    )

    return df


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(
    df: pd.DataFrame,
) -> None:
    """
    Verify that all expected columns exist.
    """

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    print(
        "✅ Required columns confirmed."
    )


# ============================================================
# DATE PROCESSING
# ============================================================

def process_dates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert Transaction_Date to datetime and remove
    rows where the date cannot be interpreted.
    """

    print(
        "\n📅 Processing transaction dates..."
    )

    df = df.copy()

    df["Transaction_Date"] = pd.to_datetime(
        df["Transaction_Date"],
        errors="coerce",
    )

    invalid_dates = int(
        df["Transaction_Date"].isna().sum()
    )

    if invalid_dates > 0:

        print(
            f"⚠️ Removing {invalid_dates} "
            "rows with invalid dates."
        )

        df = df.dropna(
            subset=["Transaction_Date"]
        )

    print(
        "✅ Date processing completed."
    )

    return df


# ============================================================
# MISSING NUMERICAL VALUES
# ============================================================

def handle_missing_numerical_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing numerical values using robust statistical
    imputation.

    Median imputation is used because it is less sensitive to
    extreme values than mean imputation.
    """

    print(
        "\n🔢 Handling missing numerical values..."
    )

    df = df.copy()

    for column in NUMERICAL_COLUMNS:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:

            print(
                f"   ✓ {column}: no missing values."
            )

            continue

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )

        print(
            f"   ✓ {column}: "
            f"{missing_before} missing values "
            f"replaced with median "
            f"{median_value:.4f}."
        )

    print(
        "✅ Numerical missing-value handling completed."
    )

    return df


# ============================================================
# MISSING CATEGORICAL VALUES
# ============================================================

def handle_missing_categorical_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing categorical values.

    Missing categorical observations are assigned the explicit
    label 'Unknown' rather than silently deleting records.
    """

    print(
        "\n🏷️ Handling missing categorical values..."
    )

    df = df.copy()

    for column in CATEGORICAL_COLUMNS:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:

            print(
                f"   ✓ {column}: no missing values."
            )

            continue

        df[column] = df[column].fillna(
            "Unknown"
        )

        print(
            f"   ✓ {column}: "
            f"{missing_before} missing values "
            "replaced with 'Unknown'."
        )

    print(
        "✅ Categorical missing-value handling completed."
    )

    return df


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove exact duplicate records.
    """

    print(
        "\n🧹 Checking duplicate records..."
    )

    df = df.copy()

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates == 0:

        print(
            "   ✓ No duplicate records found."
        )

        return df

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(
        drop=True
    )

    print(
        f"   ✓ Removed {duplicates:,} "
        "duplicate records."
    )

    return df


# ============================================================
# NUMERICAL DATA TYPE STANDARDIZATION
# ============================================================

def standardize_numeric_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize numerical columns to appropriate numeric types.
    """

    print(
        "\n🔧 Standardizing numerical data types..."
    )

    df = df.copy()

    integer_columns = [
        "Units_Sold",
        "Inventory_Level",
        "Stockout_Flag",
        "Promotion_Flag",
        "Year",
        "Month",
        "Quarter",
        "Day_of_Week",
        "Is_Weekend",
    ]

    float_columns = [
        "Unit_Price",
        "Total_Revenue",
        "Market_Index",
    ]

    for column in integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        df[column] = (
            df[column]
            .round()
            .astype("Int64")
        )

    for column in float_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        df[column] = df[column].astype(
            "float64"
        )

    print(
        "✅ Numerical data types standardized."
    )

    return df


# ============================================================
# BUSINESS RULES
# ============================================================

def apply_business_rules(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply conservative business-rule handling.

    Rules:
    ------
    1. Units_Sold cannot be negative.
    2. Unit_Price must be positive.
    3. Inventory_Level cannot be negative.
    4. Market_Index must remain between 0 and 1.
    5. Revenue must be non-negative.

    Invalid observations are removed because these values
    cannot be safely corrected without knowing the original
    source-system value.
    """

    print(
        "\n📋 Applying business rules..."
    )

    df = df.copy()

    initial_rows = len(df)

    # --------------------------------------------------------
    # Units sold
    # --------------------------------------------------------

    df = df[
        df["Units_Sold"] >= 0
    ]

    # --------------------------------------------------------
    # Unit price
    # --------------------------------------------------------

    df = df[
        df["Unit_Price"] > 0
    ]

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    df = df[
        df["Inventory_Level"] >= 0
    ]

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    df = df[
        df["Total_Revenue"] >= 0
    ]

    # --------------------------------------------------------
    # Market index
    # --------------------------------------------------------

    df = df[
        df["Market_Index"].between(
            0,
            1,
        )
    ]

    df = df.reset_index(
        drop=True
    )

    removed_rows = (
        initial_rows
        - len(df)
    )

    if removed_rows > 0:

        print(
            f"⚠️ Removed {removed_rows:,} "
            "rows that violated business rules."
        )

    else:

        print(
            "   ✓ No business-rule violations found."
        )

    print(
        "✅ Business-rule processing completed."
    )

    return df


# ============================================================
# DERIVED TIME FEATURES
# ============================================================

def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create useful calendar variables from Transaction_Date.

    These features are useful for downstream analytics and
    forecasting.
    """

    print(
        "\n🗓️ Creating time features..."
    )

    df = df.copy()

    df["Year"] = (
        df["Transaction_Date"]
        .dt.year
        .astype(int)
    )

    df["Month"] = (
        df["Transaction_Date"]
        .dt.month
        .astype(int)
    )

    df["Quarter"] = (
        df["Transaction_Date"]
        .dt.quarter
        .astype(int)
    )

    df["Day_of_Week"] = (
        df["Transaction_Date"]
        .dt.dayofweek
        .astype(int)
    )

    df["Is_Weekend"] = (
        df["Day_of_Week"] >= 5
    ).astype(int)

    # Day of month can be useful for
    # understanding transaction patterns.
    df["Day_of_Month"] = (
        df["Transaction_Date"]
        .dt.day
        .astype(int)
    )

    # Week number.
    df["Week_of_Year"] = (
        df["Transaction_Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    print(
        "✅ Time features created:"
    )

    print(
        "   - Year"
    )

    print(
        "   - Month"
    )

    print(
        "   - Quarter"
    )

    print(
        "   - Day_of_Week"
    )

    print(
        "   - Is_Weekend"
    )

    print(
        "   - Day_of_Month"
    )

    print(
        "   - Week_of_Year"
    )

    return df


# ============================================================
# REVENUE RECONCILIATION
# ============================================================

def reconcile_revenue(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recalculate revenue from Units_Sold × Unit_Price.

    A separate calculated field is retained so that the original
    reported revenue can be compared against the internally
    calculated value.
    """

    print(
        "\n💰 Reconciling revenue..."
    )

    df = df.copy()

    df["Calculated_Revenue"] = (
        df["Units_Sold"]
        * df["Unit_Price"]
    )

    df["Revenue_Difference"] = (
        df["Total_Revenue"]
        - df["Calculated_Revenue"]
    )

    difference_count = int(
        (
            df["Revenue_Difference"]
            .abs()
            > 1.0
        ).sum()
    )

    if difference_count > 0:

        print(
            f"⚠️ {difference_count:,} "
            "rows show revenue differences > ₹1."
        )

    else:

        print(
            "   ✓ Revenue reconciliation passed."
        )

    print(
        "✅ Revenue reconciliation completed."
    )

    return df


# ============================================================
# FINAL CLEANUP
# ============================================================

def final_cleanup(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform final cleanup before saving.
    """

    print(
        "\n✨ Performing final cleanup..."
    )

    df = df.copy()

    # Sort chronologically.
    df = df.sort_values(
        by="Transaction_Date"
    )

    # Reset index.
    df = df.reset_index(
        drop=True
    )

    # Ensure Market_Index remains bounded.
    df["Market_Index"] = (
        df["Market_Index"]
        .clip(
            lower=0,
            upper=1,
        )
    )

    # Round financial values.
    df["Unit_Price"] = (
        df["Unit_Price"]
        .round(2)
    )

    df["Total_Revenue"] = (
        df["Total_Revenue"]
        .round(2)
    )

    df["Calculated_Revenue"] = (
        df["Calculated_Revenue"]
        .round(2)
    )

    df["Revenue_Difference"] = (
        df["Revenue_Difference"]
        .round(2)
    )

    print(
        "✅ Final cleanup completed."
    )

    return df


# ============================================================
# PROCESSED DATA VALIDATION
# ============================================================

def validate_processed_data(
    df: pd.DataFrame,
) -> None:
    """
    Perform final checks on the processed dataset.
    """

    print(
        "\n🔍 Running final processed-data checks..."
    )

    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            "Processed dataset is empty."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = int(
        df.isna().sum().sum()
    )

    if missing_values > 0:

        raise ValueError(
            f"Processed dataset still contains "
            f"{missing_values:,} missing values."
        )

    print(
        "   ✓ No missing values."
    )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates > 0:

        raise ValueError(
            f"Processed dataset contains "
            f"{duplicates:,} duplicate rows."
        )

    print(
        "   ✓ No duplicate rows."
    )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    if (
        df["Total_Revenue"] < 0
    ).any():

        raise ValueError(
            "Negative revenue detected."
        )

    print(
        "   ✓ Revenue values are valid."
    )

    # --------------------------------------------------------
    # Market index
    # --------------------------------------------------------

    if not df["Market_Index"].between(
        0,
        1,
    ).all():

        raise ValueError(
            "Market_Index contains "
            "values outside 0–1."
        )

    print(
        "   ✓ Market_Index is within 0–1."
    )

    print(
        "✅ Final processed-data validation passed."
    )


# ============================================================
# SUMMARY
# ============================================================

def print_processing_summary(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> None:
    """
    Print a summary of the preprocessing operation.
    """

    print("\n")
    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"Original rows       : "
        f"{len(original_df):,}"
    )

    print(
        f"Processed rows      : "
        f"{len(processed_df):,}"
    )

    print(
        f"Rows removed        : "
        f"{len(original_df) - len(processed_df):,}"
    )

    print(
        f"Original columns    : "
        f"{len(original_df.columns)}"
    )

    print(
        f"Processed columns   : "
        f"{len(processed_df.columns)}"
    )

    print(
        f"Missing values      : "
        f"{processed_df.isna().sum().sum():,}"
    )

    print(
        f"Duplicate rows      : "
        f"{processed_df.duplicated().sum():,}"
    )

    print(
        f"Total revenue       : "
        f"₹{processed_df['Total_Revenue'].sum():,.2f}"
    )

    print(
        f"Average units sold  : "
        f"{processed_df['Units_Sold'].mean():,.2f}"
    )

    print("=" * 70)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_dataset() -> pd.DataFrame:
    """
    Execute the complete preprocessing workflow.
    """

    print("\n" + "=" * 70)
    print("🧹 HEALTHCARE SALES DATA PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    original_df = load_raw_data()

    # --------------------------------------------------------
    # Validate schema
    # --------------------------------------------------------

    validate_columns(
        original_df
    )

    # --------------------------------------------------------
    # Date processing
    # --------------------------------------------------------

    df = process_dates(
        original_df
    )

    # --------------------------------------------------------
    # Numerical missing values
    # --------------------------------------------------------

    df = handle_missing_numerical_values(
        df
    )

    # --------------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------------

    df = handle_missing_categorical_values(
        df
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # Standardize numeric types
    # --------------------------------------------------------

    df = standardize_numeric_types(
        df
    )

    # --------------------------------------------------------
    # Business rules
    # --------------------------------------------------------

    df = apply_business_rules(
        df
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    df = create_time_features(
        df
    )

    # --------------------------------------------------------
    # Revenue reconciliation
    # --------------------------------------------------------

    df = reconcile_revenue(
        df
    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    df = final_cleanup(
        df
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validate_processed_data(
        df
    )

    # --------------------------------------------------------
    # Create processed directory
    # --------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    df.to_csv(
        PROCESSED_DATA_FILE,
        index=False,
    )

    print(
        f"\n💾 Clean dataset saved to:"
        f"\n   {PROCESSED_DATA_FILE}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_processing_summary(
        original_df,
        df,
    )

    print(
        "\n✅ PREPROCESSING COMPLETED SUCCESSFULLY."
    )

    return df


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        preprocess_dataset()

        sys.exit(0)

    except Exception as exc:

        print(
            "\n❌ PREPROCESSING FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)