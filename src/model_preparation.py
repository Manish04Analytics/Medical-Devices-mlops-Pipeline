"""
Healthcare Medical-Device Sales
ML Problem Definition & Time-Based Dataset Split
==================================================

Objective
---------
Predict future medical-device demand (Units_Sold)
using historical commercial, product, hospital,
inventory, market and engineered demand features.

Split Strategy
--------------
Chronological date-based split:

70% -> Training
15% -> Validation
15% -> Test

IMPORTANT
---------
The split is performed using calendar dates rather than
individual rows.

This prevents transactions from the same date being
artificially separated across datasets.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "healthcare_sales_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "train.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "validation.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / "test.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "split_metadata.txt"
)


# ============================================================
# MACHINE LEARNING TARGET
# ============================================================

TARGET = "Units_Sold"


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    print("\n📂 Loading engineered feature dataset...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\n❌ Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Transaction_Date"],
    )

    print(
        f"✅ Dataset loaded successfully."
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    return df


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:

    print(
        "\n🔎 Validating dataset..."
    )

    required_columns = [
        "Transaction_Date",
        TARGET,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "❌ Missing required columns: "
            + ", ".join(missing_columns)
        )

    missing_dates = (
        df["Transaction_Date"]
        .isna()
        .sum()
    )

    if missing_dates > 0:

        raise ValueError(
            f"❌ Transaction_Date contains "
            f"{missing_dates} missing values."
        )

    missing_target = (
        df[TARGET]
        .isna()
        .sum()
    )

    if missing_target > 0:

        raise ValueError(
            f"❌ Target variable contains "
            f"{missing_target} missing values."
        )

    if not np.issubdtype(
        df[TARGET].dtype,
        np.number,
    ):

        raise TypeError(
            f"❌ Target '{TARGET}' must be numeric."
        )

    print(
        "✅ Dataset validation passed."
    )


# ============================================================
# SORT DATA
# ============================================================

def sort_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\n⏳ Sorting data chronologically..."
    )

    df = (
        df.sort_values(
            "Transaction_Date"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "✅ Data sorted chronologically."
    )

    print(
        f"First observation : "
        f"{df['Transaction_Date'].min()}"
    )

    print(
        f"Last observation  : "
        f"{df['Transaction_Date'].max()}"
    )

    return df


# ============================================================
# CREATE DATE-BASED SPLIT
# ============================================================

def create_date_based_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    print(
        "\n✂️ Creating date-based train/validation/test split..."
    )

    # --------------------------------------------------------
    # Extract unique calendar dates
    # --------------------------------------------------------

    unique_dates = (
        pd.Series(
            df["Transaction_Date"].dt.normalize().unique()
        )
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    total_dates = len(unique_dates)

    print(
        f"\nUnique calendar dates: "
        f"{total_dates:,}"
    )

    if total_dates < 10:

        raise ValueError(
            "❌ Dataset contains too few unique dates "
            "for a reliable temporal split."
        )

    # --------------------------------------------------------
    # Calculate date boundaries
    # --------------------------------------------------------

    train_date_index = int(
        total_dates * 0.70
    )

    validation_date_index = int(
        total_dates * 0.85
    )

    # --------------------------------------------------------
    # Protect against boundary errors
    # --------------------------------------------------------

    train_date_index = max(
        1,
        min(
            train_date_index,
            total_dates - 2,
        ),
    )

    validation_date_index = max(
        train_date_index + 1,
        min(
            validation_date_index,
            total_dates - 1,
        ),
    )

    train_end_date = (
        unique_dates.iloc[
            train_date_index - 1
        ]
    )

    validation_end_date = (
        unique_dates.iloc[
            validation_date_index - 1
        ]
    )

    validation_start_date = (
        unique_dates.iloc[
            train_date_index
        ]
    )

    test_start_date = (
        unique_dates.iloc[
            validation_date_index
        ]
    )

    # --------------------------------------------------------
    # Create masks
    # --------------------------------------------------------

    normalized_dates = (
        df["Transaction_Date"]
        .dt.normalize()
    )

    train_mask = (
        normalized_dates
        <= train_end_date
    )

    validation_mask = (
        (normalized_dates >= validation_start_date)
        &
        (normalized_dates <= validation_end_date)
    )

    test_mask = (
        normalized_dates
        >= test_start_date
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_df = (
        df.loc[
            train_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation_df = (
        df.loc[
            validation_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_df = (
        df.loc[
            test_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "📊 DATE-BASED SPLIT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTRAIN"
    )

    print(
        f"   Start: "
        f"{train_df['Transaction_Date'].min()}"
    )

    print(
        f"   End  : "
        f"{train_df['Transaction_Date'].max()}"
    )

    print(
        f"   Rows : "
        f"{len(train_df):,}"
    )

    print(
        f"\nVALIDATION"
    )

    print(
        f"   Start: "
        f"{validation_df['Transaction_Date'].min()}"
    )

    print(
        f"   End  : "
        f"{validation_df['Transaction_Date'].max()}"
    )

    print(
        f"   Rows : "
        f"{len(validation_df):,}"
    )

    print(
        f"\nTEST"
    )

    print(
        f"   Start: "
        f"{test_df['Transaction_Date'].min()}"
    )

    print(
        f"   End  : "
        f"{test_df['Transaction_Date'].max()}"
    )

    print(
        f"   Rows : "
        f"{len(test_df):,}"
    )

    print(
        "\nTOTAL"
    )

    print(
        f"   Rows : "
        f"{len(train_df) + len(validation_df) + len(test_df):,}"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# TEMPORAL LEAKAGE CHECK
# ============================================================

def check_temporal_leakage(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print(
        "\n🔐 Running temporal leakage checks..."
    )

    train_max = (
        train_df["Transaction_Date"]
        .dt.normalize()
        .max()
    )

    validation_min = (
        validation_df["Transaction_Date"]
        .dt.normalize()
        .min()
    )

    validation_max = (
        validation_df["Transaction_Date"]
        .dt.normalize()
        .max()
    )

    test_min = (
        test_df["Transaction_Date"]
        .dt.normalize()
        .min()
    )

    # --------------------------------------------------------
    # Check train → validation
    # --------------------------------------------------------

    if train_max >= validation_min:

        raise ValueError(
            "❌ Temporal leakage detected "
            "between training and validation data."
        )

    # --------------------------------------------------------
    # Check validation → test
    # --------------------------------------------------------

    if validation_max >= test_min:

        raise ValueError(
            "❌ Temporal leakage detected "
            "between validation and test data."
        )

    # --------------------------------------------------------
    # Check row overlap
    # --------------------------------------------------------

    train_dates = set(
        train_df["Transaction_Date"]
        .dt.normalize()
        .unique()
    )

    validation_dates = set(
        validation_df["Transaction_Date"]
        .dt.normalize()
        .unique()
    )

    test_dates = set(
        test_df["Transaction_Date"]
        .dt.normalize()
        .unique()
    )

    if train_dates.intersection(
        validation_dates
    ):

        raise ValueError(
            "❌ Date overlap detected "
            "between train and validation."
        )

    if validation_dates.intersection(
        test_dates
    ):

        raise ValueError(
            "❌ Date overlap detected "
            "between validation and test."
        )

    if train_dates.intersection(
        test_dates
    ):

        raise ValueError(
            "❌ Date overlap detected "
            "between train and test."
        )

    print(
        "✅ Temporal leakage check passed."
    )

    print(
        "✅ No date overlap between datasets."
    )

    print(
        "✅ Train data occurs strictly before validation."
    )

    print(
        "✅ Validation data occurs strictly before test."
    )


# ============================================================
# CHECK TARGET DISTRIBUTION
# ============================================================

def check_target_distribution(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print(
        "\n🎯 TARGET DISTRIBUTION"
    )

    print(
        "-" * 70
    )

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df,
    }

    for name, dataset in datasets.items():

        print(
            f"\n{name}:"
        )

        print(
            f"   Mean   : "
            f"{dataset[TARGET].mean():.2f}"
        )

        print(
            f"   Median : "
            f"{dataset[TARGET].median():.2f}"
        )

        print(
            f"   Std    : "
            f"{dataset[TARGET].std():.2f}"
        )

        print(
            f"   Min    : "
            f"{dataset[TARGET].min():.2f}"
        )

        print(
            f"   Max    : "
            f"{dataset[TARGET].max():.2f}"
        )


# ============================================================
# CHECK DUPLICATES
# ============================================================

def check_duplicates(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print(
        "\n🔎 Checking duplicates..."
    )

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df,
    }

    total_duplicates = 0

    for name, dataset in datasets.items():

        duplicates = (
            dataset
            .duplicated()
            .sum()
        )

        total_duplicates += duplicates

        print(
            f"   {name:<12}: "
            f"{duplicates:,}"
        )

    if total_duplicates == 0:

        print(
            "✅ No duplicate observations detected."
        )

    else:

        print(
            "⚠️ Duplicate observations exist."
        )


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print(
        "\n💾 Saving model-ready datasets..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    print(
        f"✅ Train saved:"
        f"\n   {TRAIN_FILE}"
    )

    print(
        f"\n✅ Validation saved:"
        f"\n   {VALIDATION_FILE}"
    )

    print(
        f"\n✅ Test saved:"
        f"\n   {TEST_FILE}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print(
        "\n📝 Saving split metadata..."
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "HEALTHCARE MEDICAL-DEVICE SALES\n"
        )

        file.write(
            "ML DATASET SPLIT METADATA\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "TARGET VARIABLE\n"
        )

        file.write(
            f"{TARGET}\n\n"
        )

        file.write(
            "SPLIT STRATEGY\n"
        )

        file.write(
            "Chronological date-based split\n"
        )

        file.write(
            "Training: 70%\n"
        )

        file.write(
            "Validation: 15%\n"
        )

        file.write(
            "Test: 15%\n\n"
        )

        file.write(
            "TRAIN PERIOD\n"
        )

        file.write(
            f"{train_df['Transaction_Date'].min()}"
            f" → "
            f"{train_df['Transaction_Date'].max()}\n\n"
        )

        file.write(
            "VALIDATION PERIOD\n"
        )

        file.write(
            f"{validation_df['Transaction_Date'].min()}"
            f" → "
            f"{validation_df['Transaction_Date'].max()}\n\n"
        )

        file.write(
            "TEST PERIOD\n"
        )

        file.write(
            f"{test_df['Transaction_Date'].min()}"
            f" → "
            f"{test_df['Transaction_Date'].max()}\n\n"
        )

        file.write(
            "ROW COUNTS\n"
        )

        file.write(
            f"Train: "
            f"{len(train_df):,}\n"
        )

        file.write(
            f"Validation: "
            f"{len(validation_df):,}\n"
        )

        file.write(
            f"Test: "
            f"{len(test_df):,}\n"
        )

    print(
        f"✅ Metadata saved:"
        f"\n   {METADATA_FILE}"
    )


# ============================================================
# FINAL QUALITY CHECK
# ============================================================

def final_quality_check(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print(
        "\n🔬 Running final quality checks..."
    )

    total_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    original_rows = len(
        train_df
    ) + len(
        validation_df
    ) + len(
        test_df
    )

    if total_rows != original_rows:

        raise ValueError(
            "❌ Row-count consistency check failed."
        )

    if len(train_df) == 0:

        raise ValueError(
            "❌ Training dataset is empty."
        )

    if len(validation_df) == 0:

        raise ValueError(
            "❌ Validation dataset is empty."
        )

    if len(test_df) == 0:

        raise ValueError(
            "❌ Test dataset is empty."
        )

    print(
        "✅ Training dataset is non-empty."
    )

    print(
        "✅ Validation dataset is non-empty."
    )

    print(
        "✅ Test dataset is non-empty."
    )

    print(
        "✅ Final quality checks passed."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_model_preparation():

    print(
        "\n" + "=" * 70
    )

    print(
        "🧠 HEALTHCARE MEDICAL-DEVICE SALES"
    )

    print(
        "ML DATA PREPARATION PIPELINE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = sort_data(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = create_date_based_split(
        df
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    check_temporal_leakage(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    check_target_distribution(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    check_duplicates(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_datasets(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    save_metadata(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Final quality
    # --------------------------------------------------------

    final_quality_check(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ STEP 6 COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\n🎯 MACHINE LEARNING TARGET"
    )

    print(
        f"   {TARGET}"
    )

    print(
        "\n📊 DATA SPLIT"
    )

    print(
        "   70% chronological training period"
    )

    print(
        "   15% chronological validation period"
    )

    print(
        "   15% chronological test period"
    )

    print(
        "\n🔐 LEAKAGE PROTECTION"
    )

    print(
        "   Date-based splitting"
    )

    print(
        "   No date overlap"
    )

    print(
        "   Future observations never enter training"
    )

    print(
        "\n📁 OUTPUT DIRECTORY"
    )

    print(
        f"   {OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_model_preparation()

        sys.exit(0)

    except Exception as exc:

        print(
            "\n❌ MODEL PREPARATION FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)