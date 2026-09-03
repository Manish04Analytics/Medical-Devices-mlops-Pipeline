"""
Healthcare Sales Data Validation
================================

Purpose
-------
Validate the raw synthetic healthcare sales dataset before it
moves into preprocessing and machine learning stages.

Validation checks include:

1. File existence
2. Required columns
3. Data types
4. Missing values
5. Duplicate records
6. Numeric range checks
7. Categorical value checks
8. Date validity
9. Revenue consistency
10. Basic business-rule validation

The validation stage acts as a quality gate.

Pipeline:
---------
Raw Data
    ↓
Data Validation
    ↓
PASS → Preprocessing
FAIL → Stop Pipeline
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


# ============================================================
# EXPECTED SCHEMA
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


EXPECTED_NUMERIC_COLUMNS = [
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


EXPECTED_CATEGORICAL_COLUMNS = [
    "Hospital_ID",
    "Hospital_Name",
    "Hospital_Type",
    "Hospital_Size",
    "Territory",
    "Product_ID",
    "Product_Name",
    "Product_Category",
]


VALID_HOSPITAL_SIZES = {
    "Small",
    "Medium",
    "Large",
}


VALID_HOSPITAL_TYPES = {
    "Tertiary_Care",
    "Multi_Specialty",
    "Specialty_Care",
}


VALID_TERRITORIES = {
    "North",
    "South",
    "East",
    "West",
    "Central",
}


VALID_PRODUCT_CATEGORIES = {
    "Radiology",
    "Patient_Monitoring",
    "Cardiology",
    "Critical_Care",
    "Gastroenterology",
}


# ============================================================
# VALIDATION RESULT CLASS
# ============================================================

class ValidationResult:
    """
    Store the overall result of the validation process.
    """

    def __init__(self) -> None:
        self.errors = []
        self.warnings = []

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file_exists(
    result: ValidationResult,
) -> bool:
    """
    Check whether the raw dataset exists.
    """

    print("\n🔎 Check 1: File existence")

    if not RAW_DATA_FILE.exists():

        result.add_error(
            f"Raw dataset not found: {RAW_DATA_FILE}"
        )

        print("❌ FAIL: Dataset file does not exist.")

        return False

    print("✅ PASS: Dataset file exists.")

    return True


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(
    result: ValidationResult,
) -> pd.DataFrame | None:
    """
    Load the raw dataset safely.
    """

    print("\n📂 Loading raw dataset...")

    try:

        df = pd.read_csv(
            RAW_DATA_FILE
        )

        print(
            f"✅ Dataset loaded successfully: "
            f"{len(df):,} rows × {len(df.columns)} columns"
        )

        return df

    except Exception as exc:

        result.add_error(
            f"Unable to load dataset: {exc}"
        )

        print(
            f"❌ FAIL: Could not load dataset: {exc}"
        )

        return None


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_required_columns(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Verify that all expected columns exist.
    """

    print("\n🔎 Check 2: Required columns")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    if missing_columns:

        result.add_error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

        print(
            "❌ FAIL: Missing columns:"
        )

        for column in missing_columns:
            print(f"   - {column}")

    else:

        print(
            "✅ PASS: All required columns are present."
        )

    if unexpected_columns:

        result.add_warning(
            "Unexpected columns found: "
            + ", ".join(unexpected_columns)
        )

        print(
            "⚠️ WARNING: Unexpected columns:"
        )

        for column in unexpected_columns:
            print(f"   - {column}")


# ============================================================
# DATA TYPE VALIDATION
# ============================================================

def validate_numeric_columns(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Verify that expected numeric columns are numeric.
    """

    print("\n🔎 Check 3: Numeric data types")

    invalid_columns = []

    for column in EXPECTED_NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            invalid_columns.append(
                column
            )

    if invalid_columns:

        result.add_error(
            "Expected numeric columns "
            "with invalid data types: "
            + ", ".join(invalid_columns)
        )

        print(
            "❌ FAIL: Invalid numeric columns:"
        )

        for column in invalid_columns:
            print(f"   - {column}")

    else:

        print(
            "✅ PASS: Numeric columns have valid types."
        )


# ============================================================
# MISSING VALUE VALIDATION
# ============================================================

def validate_missing_values(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Identify missing values.

    Missing values are treated as warnings at this stage,
    because the preprocessing stage is responsible for
    handling them.
    """

    print("\n🔎 Check 4: Missing values")

    missing_counts = (
        df.isna()
        .sum()
    )

    missing_counts = (
        missing_counts[
            missing_counts > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if missing_counts.empty:

        print(
            "✅ PASS: No missing values detected."
        )

        return

    total_missing = int(
        missing_counts.sum()
    )

    result.add_warning(
        f"{total_missing:,} missing values "
        "detected; preprocessing is required."
    )

    print(
        f"⚠️ WARNING: {total_missing:,} "
        "missing values detected."
    )

    for column, count in missing_counts.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"   - {column}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

def validate_duplicates(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Detect exact duplicate rows.
    """

    print("\n🔎 Check 5: Duplicate records")

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:

        print(
            "✅ PASS: No duplicate rows detected."
        )

        return

    result.add_error(
        f"{duplicate_count:,} duplicate "
        "rows detected."
    )

    print(
        f"❌ FAIL: {duplicate_count:,} "
        "duplicate rows detected."
    )


# ============================================================
# NUMERIC RANGE VALIDATION
# ============================================================

def validate_numeric_ranges(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Validate numerical business constraints.
    """

    print("\n🔎 Check 6: Numeric business rules")

    checks_passed = True

    # --------------------------------------------------------
    # Units sold
    # --------------------------------------------------------

    invalid_units = (
        df["Units_Sold"] < 0
    ).sum()

    if invalid_units > 0:

        checks_passed = False

        result.add_error(
            f"{invalid_units} rows contain "
            "negative Units_Sold."
        )

        print(
            f"❌ Units_Sold: "
            f"{invalid_units} invalid rows."
        )

    else:

        print(
            "✅ Units_Sold: valid."
        )

    # --------------------------------------------------------
    # Unit price
    # --------------------------------------------------------

    invalid_price = (
        df["Unit_Price"] <= 0
    ).sum()

    if invalid_price > 0:

        checks_passed = False

        result.add_error(
            f"{invalid_price} rows contain "
            "non-positive Unit_Price."
        )

        print(
            f"❌ Unit_Price: "
            f"{invalid_price} invalid rows."
        )

    else:

        print(
            "✅ Unit_Price: valid."
        )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    invalid_revenue = (
        df["Total_Revenue"] < 0
    ).sum()

    if invalid_revenue > 0:

        checks_passed = False

        result.add_error(
            f"{invalid_revenue} rows contain "
            "negative Total_Revenue."
        )

        print(
            f"❌ Total_Revenue: "
            f"{invalid_revenue} invalid rows."
        )

    else:

        print(
            "✅ Total_Revenue: valid."
        )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    invalid_inventory = (
        df["Inventory_Level"] < 0
    ).sum()

    if invalid_inventory > 0:

        checks_passed = False

        result.add_error(
            f"{invalid_inventory} rows contain "
            "negative Inventory_Level."
        )

        print(
            f"❌ Inventory_Level: "
            f"{invalid_inventory} invalid rows."
        )

    else:

        print(
            "✅ Inventory_Level: valid."
        )

    # --------------------------------------------------------
    # Market index
    # --------------------------------------------------------

    invalid_market_index = (
        (df["Market_Index"] < 0)
        | (df["Market_Index"] > 1)
    ).sum()

    if invalid_market_index > 0:

        checks_passed = False

        result.add_error(
            f"{invalid_market_index} rows contain "
            "Market_Index outside 0–1."
        )

        print(
            f"❌ Market_Index: "
            f"{invalid_market_index} invalid rows."
        )

    else:

        print(
            "✅ Market_Index: valid."
        )

    if checks_passed:

        print(
            "✅ PASS: Numeric business rules passed."
        )

    else:

        print(
            "❌ FAIL: One or more numeric "
            "business rules failed."
        )


# ============================================================
# CATEGORICAL VALIDATION
# ============================================================

def validate_categories(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Validate categorical values against the expected domain.
    """

    print("\n🔎 Check 7: Categorical values")

    category_checks = {
        "Hospital_Size": VALID_HOSPITAL_SIZES,
        "Hospital_Type": VALID_HOSPITAL_TYPES,
        "Territory": VALID_TERRITORIES,
        "Product_Category": VALID_PRODUCT_CATEGORIES,
    }

    all_valid = True

    for column, valid_values in category_checks.items():

        if column not in df.columns:
            continue

        observed_values = set(
            df[column]
            .dropna()
            .unique()
        )

        invalid_values = (
            observed_values
            - valid_values
        )

        if invalid_values:

            all_valid = False

            result.add_error(
                f"{column} contains "
                f"invalid values: "
                f"{sorted(invalid_values)}"
            )

            print(
                f"❌ {column}: "
                f"invalid values "
                f"{sorted(invalid_values)}"
            )

        else:

            print(
                f"✅ {column}: valid."
            )

    if all_valid:

        print(
            "✅ PASS: Categorical validation passed."
        )

    else:

        print(
            "❌ FAIL: Categorical validation failed."
        )


# ============================================================
# DATE VALIDATION
# ============================================================

def validate_dates(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Validate transaction dates.
    """

    print("\n🔎 Check 8: Date validation")

    try:

        dates = pd.to_datetime(
            df["Transaction_Date"],
            errors="coerce",
        )

        invalid_dates = int(
            dates.isna().sum()
        )

        if invalid_dates > 0:

            result.add_error(
                f"{invalid_dates} invalid "
                "Transaction_Date values detected."
            )

            print(
                f"❌ FAIL: {invalid_dates} "
                "invalid dates."
            )

        else:

            print(
                "✅ Date format is valid."
            )

        if not dates.empty:

            print(
                f"   Date range: "
                f"{dates.min().date()} "
                f"→ "
                f"{dates.max().date()}"
            )

    except Exception as exc:

        result.add_error(
            f"Date validation failed: {exc}"
        )

        print(
            f"❌ FAIL: Date validation error: "
            f"{exc}"
        )


# ============================================================
# REVENUE CONSISTENCY VALIDATION
# ============================================================

def validate_revenue_consistency(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Verify that revenue approximately equals:

        Units_Sold × Unit_Price

    A small numerical tolerance is allowed.
    """

    print("\n🔎 Check 9: Revenue consistency")

    calculated_revenue = (
        df["Units_Sold"]
        * df["Unit_Price"]
    )

    difference = np.abs(
        calculated_revenue
        - df["Total_Revenue"]
    )

    tolerance = 1.0

    invalid_rows = (
        difference > tolerance
    ).sum()

    if invalid_rows > 0:

        result.add_error(
            f"{invalid_rows:,} rows have "
            "inconsistent revenue calculations."
        )

        print(
            f"❌ FAIL: {invalid_rows:,} "
            "inconsistent revenue rows."
        )

    else:

        print(
            "✅ PASS: Revenue calculations "
            "are internally consistent."
        )


# ============================================================
# FLAG VALIDATION
# ============================================================

def validate_binary_flags(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Validate binary indicator columns.
    """

    print("\n🔎 Check 10: Binary flags")

    flag_columns = [
        "Stockout_Flag",
        "Promotion_Flag",
        "Is_Weekend",
    ]

    all_valid = True

    for column in flag_columns:

        if column not in df.columns:
            continue

        observed_values = set(
            df[column]
            .dropna()
            .unique()
        )

        if not observed_values.issubset(
            {0, 1}
        ):

            all_valid = False

            result.add_error(
                f"{column} contains values "
                f"outside {{0, 1}}."
            )

            print(
                f"❌ {column}: invalid values "
                f"{sorted(observed_values)}"
            )

        else:

            print(
                f"✅ {column}: valid."
            )

    if all_valid:

        print(
            "✅ PASS: Binary flag validation passed."
        )

    else:

        print(
            "❌ FAIL: Binary flag validation failed."
        )


# ============================================================
# DATA QUALITY REPORT
# ============================================================

def print_quality_summary(
    df: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """
    Print final validation summary.
    """

    print("\n")
    print("=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)

    print(
        f"Rows                : {len(df):,}"
    )

    print(
        f"Columns             : {len(df.columns)}"
    )

    print(
        f"Duplicate rows      : "
        f"{df.duplicated().sum():,}"
    )

    print(
        f"Missing values      : "
        f"{df.isna().sum().sum():,}"
    )

    print(
        f"Validation errors   : "
        f"{len(result.errors)}"
    )

    print(
        f"Validation warnings : "
        f"{len(result.warnings)}"
    )

    print("=" * 70)

    if result.warnings:

        print("\n⚠️ WARNINGS:")

        for warning in result.warnings:

            print(
                f"   • {warning}"
            )

    if result.errors:

        print("\n❌ ERRORS:")

        for error in result.errors:

            print(
                f"   • {error}"
            )

    print("\n" + "=" * 70)

    if result.passed:

        print(
            "✅ DATA VALIDATION PASSED"
        )

        print(
            "The dataset can proceed to preprocessing."
        )

    else:

        print(
            "❌ DATA VALIDATION FAILED"
        )

        print(
            "The pipeline must stop until "
            "critical data-quality errors are fixed."
        )

    print("=" * 70)


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validate_dataset() -> bool:
    """
    Run the complete data-validation pipeline.

    Returns
    -------
    bool
        True if no critical errors were detected.
        False otherwise.
    """

    print("\n" + "=" * 70)
    print("🔍 HEALTHCARE SALES DATA VALIDATION")
    print("=" * 70)

    result = ValidationResult()

    # --------------------------------------------------------
    # File check
    # --------------------------------------------------------

    file_exists = validate_file_exists(
        result
    )

    if not file_exists:

        print_quality_summary(
            pd.DataFrame(),
            result,
        )

        return False

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_dataset(
        result
    )

    if df is None:

        print_quality_summary(
            pd.DataFrame(),
            result,
        )

        return False

    # --------------------------------------------------------
    # Run validation checks
    # --------------------------------------------------------

    validate_required_columns(
        df,
        result,
    )

    validate_numeric_columns(
        df,
        result,
    )

    validate_missing_values(
        df,
        result,
    )

    validate_duplicates(
        df,
        result,
    )

    validate_numeric_ranges(
        df,
        result,
    )

    validate_categories(
        df,
        result,
    )

    validate_dates(
        df,
        result,
    )

    validate_revenue_consistency(
        df,
        result,
    )

    validate_binary_flags(
        df,
        result,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_quality_summary(
        df,
        result,
    )

    return result.passed


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    validation_passed = validate_dataset()

    if validation_passed:

        sys.exit(0)

    else:

        sys.exit(1)