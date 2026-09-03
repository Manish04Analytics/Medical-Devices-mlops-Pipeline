"""
Synthetic Healthcare Sales Data Generator
==========================================

Purpose
-------
Generate a realistic synthetic healthcare sales dataset for the
Medical Device Demand Intelligence / MLOps project.

IMPORTANT
---------
This dataset is SYNTHETIC and does not represent real hospital,
patient, physician, or commercial data.

The generator creates reproducible transactional sales data containing:

- Transaction dates
- Hospital accounts
- Hospital types
- Territories
- Product categories
- Product names
- Units sold
- Unit prices
- Revenue
- Inventory levels
- Market index
- Promotional activity
- Seasonal effects

The generated data is designed for:

1. Data validation
2. Exploratory data analysis
3. Feature engineering
4. Demand forecasting
5. Machine learning
6. MLOps experimentation
7. Data drift / monitoring demonstrations

Output
------
data/raw/healthcare_sales.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Number of synthetic transactions.
N_RECORDS = 10_000

# Date range for synthetic historical data.
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# Output location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "healthcare_sales.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)


# ============================================================
# MASTER DATA
# ============================================================

HOSPITALS = {
    "H001": {
        "name": "Hospital_A",
        "type": "Tertiary_Care",
        "size": "Large",
        "territory": "North",
    },
    "H002": {
        "name": "Hospital_B",
        "type": "Multi_Specialty",
        "size": "Large",
        "territory": "North",
    },
    "H003": {
        "name": "Hospital_C",
        "type": "Tertiary_Care",
        "size": "Medium",
        "territory": "West",
    },
    "H004": {
        "name": "Hospital_D",
        "type": "Multi_Specialty",
        "size": "Medium",
        "territory": "West",
    },
    "H005": {
        "name": "Hospital_E",
        "type": "Specialty_Care",
        "size": "Medium",
        "territory": "South",
    },
    "H006": {
        "name": "Hospital_F",
        "type": "Tertiary_Care",
        "size": "Large",
        "territory": "South",
    },
    "H007": {
        "name": "Hospital_G",
        "type": "Multi_Specialty",
        "size": "Large",
        "territory": "East",
    },
    "H008": {
        "name": "Hospital_H",
        "type": "Specialty_Care",
        "size": "Small",
        "territory": "East",
    },
    "H009": {
        "name": "Hospital_I",
        "type": "Multi_Specialty",
        "size": "Medium",
        "territory": "Central",
    },
    "H010": {
        "name": "Hospital_J",
        "type": "Tertiary_Care",
        "size": "Large",
        "territory": "Central",
    },
}


PRODUCTS = {
    "P001": {
        "name": "Imaging_System",
        "category": "Radiology",
        "base_price": 1_800_000,
    },
    "P002": {
        "name": "Ultrasound_System",
        "category": "Radiology",
        "base_price": 750_000,
    },
    "P003": {
        "name": "Patient_Monitor",
        "category": "Patient_Monitoring",
        "base_price": 125_000,
    },
    "P004": {
        "name": "ECG_System",
        "category": "Cardiology",
        "base_price": 85_000,
    },
    "P005": {
        "name": "Cardiac_Monitor",
        "category": "Cardiology",
        "base_price": 150_000,
    },
    "P006": {
        "name": "Infusion_Pump",
        "category": "Critical_Care",
        "base_price": 65_000,
    },
    "P007": {
        "name": "Ventilator",
        "category": "Critical_Care",
        "base_price": 350_000,
    },
    "P008": {
        "name": "Endoscopy_System",
        "category": "Gastroenterology",
        "base_price": 1_100_000,
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_date_series(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Create a daily date range between start and end dates.
    """
    return pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )


def get_seasonality(month: int) -> float:
    """
    Return a monthly seasonality multiplier.

    The values intentionally create moderate seasonal variation
    rather than extreme artificial patterns.
    """
    seasonal_effects = {
        1: 0.95,
        2: 0.97,
        3: 1.02,
        4: 1.05,
        5: 1.08,
        6: 1.03,
        7: 0.98,
        8: 1.00,
        9: 1.06,
        10: 1.10,
        11: 1.07,
        12: 0.93,
    }

    return seasonal_effects.get(month, 1.0)


def get_hospital_multiplier(hospital_size: str) -> float:
    """
    Return expected demand multiplier based on hospital size.
    """
    size_multipliers = {
        "Small": 0.70,
        "Medium": 1.00,
        "Large": 1.35,
    }

    return size_multipliers.get(hospital_size, 1.0)


def get_product_multiplier(product_category: str) -> float:
    """
    Return expected demand multiplier based on product category.
    """
    category_multipliers = {
        "Radiology": 0.75,
        "Patient_Monitoring": 1.30,
        "Cardiology": 1.10,
        "Critical_Care": 1.20,
        "Gastroenterology": 0.80,
    }

    return category_multipliers.get(product_category, 1.0)


# ============================================================
# MAIN DATA GENERATION FUNCTION
# ============================================================

def generate_dataset() -> pd.DataFrame:
    """
    Generate the complete synthetic healthcare sales dataset.
    """

    dates = create_date_series(
        START_DATE,
        END_DATE,
    )

    hospital_ids = list(HOSPITALS.keys())
    product_ids = list(PRODUCTS.keys())

    rows = []

    for _ in range(N_RECORDS):

        # ----------------------------------------------------
        # Random transaction date
        # ----------------------------------------------------

        transaction_date = np.random.choice(dates)

        # Convert numpy datetime to pandas Timestamp.
        transaction_date = pd.Timestamp(transaction_date)

        # ----------------------------------------------------
        # Select hospital
        # ----------------------------------------------------

        hospital_id = np.random.choice(hospital_ids)

        hospital = HOSPITALS[hospital_id]

        # ----------------------------------------------------
        # Select product
        # ----------------------------------------------------

        product_id = np.random.choice(product_ids)

        product = PRODUCTS[product_id]

        # ----------------------------------------------------
        # Calendar variables
        # ----------------------------------------------------

        year = transaction_date.year
        month = transaction_date.month
        quarter = transaction_date.quarter
        day_of_week = transaction_date.dayofweek

        is_weekend = int(day_of_week >= 5)

        # ----------------------------------------------------
        # Seasonality
        # ----------------------------------------------------

        seasonal_multiplier = get_seasonality(month)

        # ----------------------------------------------------
        # Hospital demand effect
        # ----------------------------------------------------

        hospital_multiplier = get_hospital_multiplier(
            hospital["size"]
        )

        # ----------------------------------------------------
        # Product demand effect
        # ----------------------------------------------------

        product_multiplier = get_product_multiplier(
            product["category"]
        )

        # ----------------------------------------------------
        # Long-term market growth
        # ----------------------------------------------------

        days_since_start = (
            transaction_date - pd.Timestamp(START_DATE)
        ).days

        market_growth = 1 + (
            0.00025 * days_since_start
        )

        # ----------------------------------------------------
        # Market index
        # ----------------------------------------------------

        market_index = np.clip(
            0.50
            + 0.08 * np.sin(month / 12 * 2 * np.pi)
            + 0.00003 * days_since_start
            + np.random.normal(0, 0.035),
            0.10,
            0.95,
        )

        # ----------------------------------------------------
        # Promotional activity
        # ----------------------------------------------------

        promotion_probability = 0.18

        is_promotion = int(
            np.random.random() < promotion_probability
        )

        promotion_multiplier = (
            1.12 if is_promotion else 1.00
        )

        # ----------------------------------------------------
        # Base demand
        # ----------------------------------------------------

        base_demand = (
            20
            * hospital_multiplier
            * product_multiplier
            * seasonal_multiplier
            * market_growth
            * promotion_multiplier
        )

        # ----------------------------------------------------
        # Random demand variation
        # ----------------------------------------------------

        demand_noise = np.random.normal(
            loc=0,
            scale=3.0,
        )

        units_sold = max(
            1,
            int(round(base_demand + demand_noise)),
        )

        # ----------------------------------------------------
        # Unit price variation
        # ----------------------------------------------------

        price_variation = np.random.normal(
            loc=1.0,
            scale=0.04,
        )

        unit_price = max(
            1,
            product["base_price"] * price_variation,
        )

        unit_price = round(
            unit_price,
            2,
        )

        # ----------------------------------------------------
        # Revenue
        # ----------------------------------------------------

        total_revenue = (
            units_sold * unit_price
        )

        total_revenue = round(
            total_revenue,
            2,
        )

        # ----------------------------------------------------
        # Inventory
        # ----------------------------------------------------

        inventory_level = max(
            0,
            int(
                units_sold
                * np.random.uniform(1.5, 4.0)
                + np.random.normal(0, 5)
            ),
        )

        # ----------------------------------------------------
        # Stockout indicator
        # ----------------------------------------------------

        stockout_flag = int(
            inventory_level < units_sold
        )

        # ----------------------------------------------------
        # Territory
        # ----------------------------------------------------

        territory = hospital["territory"]

        # ----------------------------------------------------
        # Store row
        # ----------------------------------------------------

        rows.append(
            {
                "Transaction_Date": transaction_date,
                "Hospital_ID": hospital_id,
                "Hospital_Name": hospital["name"],
                "Hospital_Type": hospital["type"],
                "Hospital_Size": hospital["size"],
                "Territory": territory,
                "Product_ID": product_id,
                "Product_Name": product["name"],
                "Product_Category": product["category"],
                "Units_Sold": units_sold,
                "Unit_Price": unit_price,
                "Total_Revenue": total_revenue,
                "Inventory_Level": inventory_level,
                "Stockout_Flag": stockout_flag,
                "Market_Index": round(
                    market_index,
                    4,
                ),
                "Promotion_Flag": is_promotion,
                "Year": year,
                "Month": month,
                "Quarter": quarter,
                "Day_of_Week": day_of_week,
                "Is_Weekend": is_weekend,
            }
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        by="Transaction_Date"
    ).reset_index(drop=True)

    return df


# ============================================================
# DATA QUALITY SCENARIO
# ============================================================

def introduce_controlled_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Introduce a very small number of missing values.

    Purpose:
    --------
    This intentionally creates a realistic data-quality scenario
    so that the downstream validation and preprocessing stages
    can demonstrate how missing values are detected and handled.

    NOTE:
    -----
    Missing values are introduced only into selected numerical
    columns and represent less than 1% of the dataset.
    """

    df = df.copy()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    columns = [
        "Unit_Price",
        "Inventory_Level",
        "Market_Index",
    ]

    missing_fraction = 0.005

    n_missing = max(
        1,
        int(len(df) * missing_fraction),
    )

    for column in columns:

        missing_indices = rng.choice(
            df.index,
            size=n_missing,
            replace=False,
        )

        df.loc[
            missing_indices,
            column,
        ] = np.nan

    return df


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def print_dataset_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a concise summary of the generated dataset.
    """

    print("\n" + "=" * 70)
    print("SYNTHETIC HEALTHCARE SALES DATASET")
    print("=" * 70)

    print(
        f"Rows              : {len(df):,}"
    )

    print(
        f"Columns           : {df.shape[1]}"
    )

    print(
        f"Date range        : "
        f"{df['Transaction_Date'].min().date()} "
        f"to "
        f"{df['Transaction_Date'].max().date()}"
    )

    print(
        f"Hospitals         : "
        f"{df['Hospital_ID'].nunique()}"
    )

    print(
        f"Products          : "
        f"{df['Product_ID'].nunique()}"
    )

    print(
        f"Territories       : "
        f"{df['Territory'].nunique()}"
    )

    print(
        f"Total revenue     : "
        f"₹{df['Total_Revenue'].sum():,.2f}"
    )

    print(
        f"Average units     : "
        f"{df['Units_Sold'].mean():,.2f}"
    )

    print(
        f"Missing values    : "
        f"{df.isna().sum().sum():,}"
    )

    print(
        f"Duplicate rows    : "
        f"{df.duplicated().sum():,}"
    )

    print("=" * 70)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main() -> None:
    """
    Generate and save the synthetic dataset.
    """

    print(
        "\n🚀 Starting synthetic healthcare "
        "sales data generation..."
    )

    print(
        f"📊 Target records: {N_RECORDS:,}"
    )

    print(
        f"📅 Date range: "
        f"{START_DATE} → {END_DATE}"
    )

    # --------------------------------------------------------
    # Generate base dataset
    # --------------------------------------------------------

    df = generate_dataset()

    print(
        "✅ Base transactional dataset generated."
    )

    # --------------------------------------------------------
    # Introduce controlled missing values
    # --------------------------------------------------------

    df = introduce_controlled_missing_values(
        df
    )

    print(
        "⚠️ Controlled missing values introduced "
        "for validation testing."
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"💾 Dataset saved to:\n"
        f"   {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_dataset_summary(
        df
    )

    print(
        "\n✅ Data generation completed successfully."
    )

    print(
        "ℹ️ IMPORTANT: This is synthetic data "
        "created for portfolio/MLOps experimentation."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()