"""
Healthcare Medical-Device Sales
Exploratory Data Analysis & Business Insights
================================================

Purpose
-------
Analyze the engineered healthcare sales dataset before machine
learning.

The analysis focuses on:

1. Revenue performance
2. Unit demand
3. Hospital-account performance
4. Product performance
5. Territory performance
6. Monthly demand trends
7. Quarterly performance
8. Inventory and stockout behaviour
9. Promotion behaviour
10. Demand volatility
11. Feature correlations
12. Automated business insights

Input
-----
data/processed/healthcare_sales_features.csv

Outputs
-------
visualizations/
    revenue_by_hospital.png
    revenue_by_product_category.png
    revenue_by_territory.png
    monthly_revenue_trend.png
    monthly_demand_trend.png
    quarterly_revenue.png
    inventory_vs_demand.png
    stockout_analysis.png
    promotion_analysis.png
    demand_distribution.png
    demand_volatility.png
    correlation_heatmap.png

reports/
    eda_summary.txt
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "healthcare_sales_features.csv"
)

VISUALIZATION_DIR = (
    PROJECT_ROOT
    / "visualizations"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "eda_summary.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load the engineered feature dataset.
    """

    print("\n📂 Loading engineered dataset...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n{INPUT_FILE}"
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
# DIRECTORY SETUP
# ============================================================

def create_output_directories() -> None:
    """
    Create directories for charts and reports.
    """

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "📁 Output directories ready."
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

def generate_dataset_overview(
    df: pd.DataFrame,
) -> dict:
    """
    Generate high-level dataset statistics.
    """

    print("\n" + "=" * 70)
    print("📊 DATASET OVERVIEW")
    print("=" * 70)

    overview = {}

    overview["rows"] = len(df)

    overview["columns"] = len(df.columns)

    overview["hospitals"] = (
        df["Hospital_ID"]
        .nunique()
    )

    overview["products"] = (
        df["Product_ID"]
        .nunique()
    )

    overview["categories"] = (
        df["Product_Category"]
        .nunique()
    )

    overview["territories"] = (
        df["Territory"]
        .nunique()
    )

    overview["total_units"] = (
        df["Units_Sold"]
        .sum()
    )

    overview["total_revenue"] = (
        df["Total_Revenue"]
        .sum()
    )

    overview["average_transaction_revenue"] = (
        df["Total_Revenue"]
        .mean()
    )

    overview["median_transaction_revenue"] = (
        df["Total_Revenue"]
        .median()
    )

    overview["date_start"] = (
        df["Transaction_Date"]
        .min()
    )

    overview["date_end"] = (
        df["Transaction_Date"]
        .max()
    )

    print(
        f"Rows                 : {overview['rows']:,}"
    )

    print(
        f"Columns              : {overview['columns']:,}"
    )

    print(
        f"Hospitals            : {overview['hospitals']:,}"
    )

    print(
        f"Products             : {overview['products']:,}"
    )

    print(
        f"Product categories   : {overview['categories']:,}"
    )

    print(
        f"Territories          : {overview['territories']:,}"
    )

    print(
        f"Total units sold     : "
        f"{overview['total_units']:,.0f}"
    )

    print(
        f"Total revenue        : "
        f"₹{overview['total_revenue']:,.2f}"
    )

    print(
        f"Average transaction  : "
        f"₹{overview['average_transaction_revenue']:,.2f}"
    )

    print(
        f"Median transaction   : "
        f"₹{overview['median_transaction_revenue']:,.2f}"
    )

    print(
        f"Date range           : "
        f"{overview['date_start'].date()} "
        f"→ "
        f"{overview['date_end'].date()}"
    )

    return overview


# ============================================================
# REVENUE BY HOSPITAL
# ============================================================

def analyze_revenue_by_hospital(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze revenue concentration across hospital accounts.
    """

    print("\n🏥 Analyzing revenue by hospital...")

    result = (
        df.groupby(
            [
                "Hospital_ID",
                "Hospital_Name",
            ],
            as_index=False,
        )
        .agg(
            Total_Revenue=(
                "Total_Revenue",
                "sum",
            ),
            Units_Sold=(
                "Units_Sold",
                "sum",
            ),
            Transactions=(
                "Product_ID",
                "count",
            ),
        )
        .sort_values(
            "Total_Revenue",
            ascending=False,
        )
    )

    result["Revenue_Share"] = (
        result["Total_Revenue"]
        / result["Total_Revenue"].sum()
    )

    result["Revenue_Share_Pct"] = (
        result["Revenue_Share"]
        * 100
    )

    print(
        "\nTop hospital accounts:"
    )

    for _, row in result.head(10).iterrows():

        print(
            f"   {row['Hospital_Name']:<30}"
            f" ₹{row['Total_Revenue']:,.0f}"
            f" ({row['Revenue_Share_Pct']:.2f}%)"
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plot_data = result.head(10).sort_values(
        "Total_Revenue"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        plot_data["Hospital_Name"],
        plot_data["Total_Revenue"],
    )

    plt.title(
        "Top Hospital Accounts by Revenue"
    )

    plt.xlabel(
        "Total Revenue (₹)"
    )

    plt.ylabel(
        "Hospital"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "revenue_by_hospital.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: revenue_by_hospital.png"
    )

    return result


# ============================================================
# PRODUCT CATEGORY ANALYSIS
# ============================================================

def analyze_product_categories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze revenue and demand by product category.
    """

    print(
        "\n🔬 Analyzing product categories..."
    )

    result = (
        df.groupby(
            "Product_Category",
            as_index=False,
        )
        .agg(
            Total_Revenue=(
                "Total_Revenue",
                "sum",
            ),
            Units_Sold=(
                "Units_Sold",
                "sum",
            ),
            Average_Price=(
                "Unit_Price",
                "mean",
            ),
        )
        .sort_values(
            "Total_Revenue",
            ascending=False,
        )
    )

    result["Revenue_Share_Pct"] = (
        result["Total_Revenue"]
        / result["Total_Revenue"].sum()
        * 100
    )

    print(
        "\nProduct category performance:"
    )

    for _, row in result.iterrows():

        print(
            f"   {row['Product_Category']:<25}"
            f" Revenue: ₹{row['Total_Revenue']:,.0f}"
            f" | Units: {row['Units_Sold']:,.0f}"
            f" | Share: {row['Revenue_Share_Pct']:.2f}%"
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        result["Product_Category"],
        result["Total_Revenue"],
    )

    plt.title(
        "Revenue by Product Category"
    )

    plt.xlabel(
        "Product Category"
    )

    plt.ylabel(
        "Revenue (₹)"
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "revenue_by_product_category.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: revenue_by_product_category.png"
    )

    return result


# ============================================================
# TERRITORY ANALYSIS
# ============================================================

def analyze_territories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze territory-level commercial performance.
    """

    print(
        "\n🗺️ Analyzing territories..."
    )

    result = (
        df.groupby(
            "Territory",
            as_index=False,
        )
        .agg(
            Total_Revenue=(
                "Total_Revenue",
                "sum",
            ),
            Units_Sold=(
                "Units_Sold",
                "sum",
            ),
            Hospital_Count=(
                "Hospital_ID",
                "nunique",
            ),
        )
        .sort_values(
            "Total_Revenue",
            ascending=False,
        )
    )

    result["Revenue_Share_Pct"] = (
        result["Total_Revenue"]
        / result["Total_Revenue"].sum()
        * 100
    )

    print(
        "\nTerritory performance:"
    )

    for _, row in result.iterrows():

        print(
            f"   {row['Territory']:<15}"
            f" Revenue: ₹{row['Total_Revenue']:,.0f}"
            f" | Hospitals: {row['Hospital_Count']}"
            f" | Share: {row['Revenue_Share_Pct']:.2f}%"
        )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        result["Territory"],
        result["Total_Revenue"],
    )

    plt.title(
        "Revenue by Territory"
    )

    plt.xlabel(
        "Territory"
    )

    plt.ylabel(
        "Revenue (₹)"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "revenue_by_territory.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: revenue_by_territory.png"
    )

    return result


# ============================================================
# MONTHLY TREND ANALYSIS
# ============================================================

def analyze_monthly_trends(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze monthly revenue and demand trends.
    """

    print(
        "\n📅 Analyzing monthly trends..."
    )

    monthly = (
        df.groupby(
            pd.Grouper(
                key="Transaction_Date",
                freq="MS",
            )
        )
        .agg(
            Total_Revenue=(
                "Total_Revenue",
                "sum",
            ),
            Units_Sold=(
                "Units_Sold",
                "sum",
            ),
            Transactions=(
                "Product_ID",
                "count",
            ),
        )
        .reset_index()
    )

    monthly["Revenue_Growth_Pct"] = (
        monthly["Total_Revenue"]
        .pct_change()
        * 100
    )

    monthly["Demand_Growth_Pct"] = (
        monthly["Units_Sold"]
        .pct_change()
        * 100
    )

    print(
        "\nMonthly trend:"
    )

    for _, row in monthly.iterrows():

        print(
            f"   {row['Transaction_Date'].strftime('%Y-%m')}"
            f" | Revenue ₹{row['Total_Revenue']:,.0f}"
            f" | Units {row['Units_Sold']:,.0f}"
        )

    # --------------------------------------------------------
    # Revenue chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        monthly["Transaction_Date"],
        monthly["Total_Revenue"],
        marker="o",
    )

    plt.title(
        "Monthly Revenue Trend"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Revenue (₹)"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "monthly_revenue_trend.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Demand chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        monthly["Transaction_Date"],
        monthly["Units_Sold"],
        marker="o",
    )

    plt.title(
        "Monthly Unit Demand Trend"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Units Sold"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "monthly_demand_trend.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved monthly trend visualizations."
    )

    return monthly, monthly.copy()


# ============================================================
# QUARTERLY ANALYSIS
# ============================================================

def analyze_quarters(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze quarterly revenue and demand.
    """

    print(
        "\n📆 Analyzing quarterly performance..."
    )

    result = (
        df.groupby(
            [
                "Year",
                "Quarter",
            ],
            as_index=False,
        )
        .agg(
            Total_Revenue=(
                "Total_Revenue",
                "sum",
            ),
            Units_Sold=(
                "Units_Sold",
                "sum",
            ),
        )
    )

    result["Quarter_Label"] = (
        result["Year"].astype(str)
        + "-Q"
        + result["Quarter"].astype(str)
    )

    print(
        "\nQuarterly performance:"
    )

    for _, row in result.iterrows():

        print(
            f"   {row['Quarter_Label']}"
            f" | Revenue ₹{row['Total_Revenue']:,.0f}"
            f" | Units {row['Units_Sold']:,.0f}"
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        result["Quarter_Label"],
        result["Total_Revenue"],
    )

    plt.title(
        "Quarterly Revenue Performance"
    )

    plt.xlabel(
        "Quarter"
    )

    plt.ylabel(
        "Revenue (₹)"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "quarterly_revenue.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: quarterly_revenue.png"
    )

    return result


# ============================================================
# INVENTORY VS DEMAND
# ============================================================

def analyze_inventory(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze inventory levels relative to demand.
    """

    print(
        "\n📦 Analyzing inventory behaviour..."
    )

    result = {}

    result["average_inventory"] = (
        df["Inventory_Level"]
        .mean()
    )

    result["median_inventory"] = (
        df["Inventory_Level"]
        .median()
    )

    result["average_demand"] = (
        df["Units_Sold"]
        .mean()
    )

    result["low_inventory_rate"] = (
        df["Low_Inventory_Flag"]
        .mean()
        * 100
    )

    result["high_inventory_rate"] = (
        df["High_Inventory_Flag"]
        .mean()
        * 100
    )

    print(
        f"   Average inventory: "
        f"{result['average_inventory']:,.2f}"
    )

    print(
        f"   Median inventory: "
        f"{result['median_inventory']:,.2f}"
    )

    print(
        f"   Average demand: "
        f"{result['average_demand']:,.2f}"
    )

    print(
        f"   Low-inventory rate: "
        f"{result['low_inventory_rate']:.2f}%"
    )

    print(
        f"   High-inventory rate: "
        f"{result['high_inventory_rate']:.2f}%"
    )

    # --------------------------------------------------------
    # Scatter plot
    # --------------------------------------------------------

    sample = df.sample(
        n=min(3000, len(df)),
        random_state=42,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        sample["Inventory_Level"],
        sample["Units_Sold"],
        alpha=0.4,
    )

    plt.title(
        "Inventory Level vs Unit Demand"
    )

    plt.xlabel(
        "Inventory Level"
    )

    plt.ylabel(
        "Units Sold"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "inventory_vs_demand.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: inventory_vs_demand.png"
    )

    return result


# ============================================================
# STOCKOUT ANALYSIS
# ============================================================

def analyze_stockouts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare demand and revenue under stockout versus
    non-stockout conditions.
    """

    print(
        "\n🚨 Analyzing stockout behaviour..."
    )

    result = (
        df.groupby(
            "Stockout_Flag",
            as_index=False,
        )
        .agg(
            Average_Units_Sold=(
                "Units_Sold",
                "mean",
            ),
            Average_Revenue=(
                "Total_Revenue",
                "mean",
            ),
            Transactions=(
                "Product_ID",
                "count",
            ),
        )
    )

    result["Stockout_Status"] = (
        result["Stockout_Flag"]
        .map(
            {
                0: "No Stockout",
                1: "Stockout",
            }
        )
    )

    print(
        "\nStockout comparison:"
    )

    for _, row in result.iterrows():

        print(
            f"   {row['Stockout_Status']:<15}"
            f" | Avg Units: {row['Average_Units_Sold']:.2f}"
            f" | Avg Revenue: ₹{row['Average_Revenue']:,.2f}"
        )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        result["Stockout_Status"],
        result["Average_Units_Sold"],
    )

    plt.title(
        "Average Demand by Stockout Status"
    )

    plt.xlabel(
        "Stockout Status"
    )

    plt.ylabel(
        "Average Units Sold"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "stockout_analysis.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: stockout_analysis.png"
    )

    return result


# ============================================================
# PROMOTION ANALYSIS
# ============================================================

def analyze_promotions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare sales behaviour during promotion versus
    non-promotion observations.
    """

    print(
        "\n📣 Analyzing promotion behaviour..."
    )

    result = (
        df.groupby(
            "Promotion_Flag",
            as_index=False,
        )
        .agg(
            Average_Units_Sold=(
                "Units_Sold",
                "mean",
            ),
            Average_Revenue=(
                "Total_Revenue",
                "mean",
            ),
            Transactions=(
                "Product_ID",
                "count",
            ),
        )
    )

    result["Promotion_Status"] = (
        result["Promotion_Flag"]
        .map(
            {
                0: "No Promotion",
                1: "Promotion",
            }
        )
    )

    print(
        "\nPromotion comparison:"
    )

    for _, row in result.iterrows():

        print(
            f"   {row['Promotion_Status']:<15}"
            f" | Avg Units: {row['Average_Units_Sold']:.2f}"
            f" | Avg Revenue: ₹{row['Average_Revenue']:,.2f}"
        )

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        result["Promotion_Status"],
        result["Average_Units_Sold"],
    )

    plt.title(
        "Average Demand by Promotion Status"
    )

    plt.xlabel(
        "Promotion Status"
    )

    plt.ylabel(
        "Average Units Sold"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "promotion_analysis.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: promotion_analysis.png"
    )

    return result


# ============================================================
# DEMAND DISTRIBUTION
# ============================================================

def analyze_demand_distribution(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze the distribution and skewness of demand.
    """

    print(
        "\n📊 Analyzing demand distribution..."
    )

    demand = df["Units_Sold"]

    result = {
        "mean": demand.mean(),
        "median": demand.median(),
        "std": demand.std(),
        "min": demand.min(),
        "max": demand.max(),
        "skewness": demand.skew(),
    }

    print(
        f"   Mean demand       : {result['mean']:.2f}"
    )

    print(
        f"   Median demand     : {result['median']:.2f}"
    )

    print(
        f"   Standard deviation: {result['std']:.2f}"
    )

    print(
        f"   Minimum demand    : {result['min']:.2f}"
    )

    print(
        f"   Maximum demand    : {result['max']:.2f}"
    )

    print(
        f"   Skewness          : {result['skewness']:.2f}"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        demand,
        bins=40,
    )

    plt.title(
        "Distribution of Unit Demand"
    )

    plt.xlabel(
        "Units Sold"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "demand_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: demand_distribution.png"
    )

    return result


# ============================================================
# DEMAND VOLATILITY
# ============================================================

def analyze_demand_volatility(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze demand volatility by product category.
    """

    print(
        "\n📉 Analyzing demand volatility..."
    )

    result = (
        df.groupby(
            "Product_Category",
            as_index=False,
        )
        .agg(
            Average_CV=(
                "Demand_CV_28D",
                "mean",
            ),
            Volatile_Observation_Rate=(
                "Demand_Volatility_Flag",
                "mean",
            ),
        )
        .sort_values(
            "Average_CV",
            ascending=False,
        )
    )

    result["Volatile_Observation_Rate"] *= 100

    print(
        "\nDemand volatility by category:"
    )

    for _, row in result.iterrows():

        print(
            f"   {row['Product_Category']:<25}"
            f" CV: {row['Average_CV']:.3f}"
            f" | Volatile: "
            f"{row['Volatile_Observation_Rate']:.2f}%"
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        result["Product_Category"],
        result["Average_CV"],
    )

    plt.title(
        "Demand Volatility by Product Category"
    )

    plt.xlabel(
        "Product Category"
    )

    plt.ylabel(
        "Average 28-Day Coefficient of Variation"
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "demand_volatility.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: demand_volatility.png"
    )

    return result


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate correlations among major numerical variables.
    """

    print(
        "\n🔗 Analyzing numerical correlations..."
    )

    correlation_columns = [
        "Units_Sold",
        "Unit_Price",
        "Total_Revenue",
        "Inventory_Level",
        "Stockout_Flag",
        "Market_Index",
        "Promotion_Flag",
        "Demand_Lag_7",
        "Demand_Lag_28",
        "Demand_Rolling_Mean_7",
        "Demand_Rolling_Mean_28",
        "Demand_Growth_7D",
        "Demand_Growth_28D",
        "Inventory_Coverage",
        "Demand_CV_28D",
    ]

    available_columns = [
        column
        for column in correlation_columns
        if column in df.columns
    ]

    correlation = (
        df[available_columns]
        .corr()
    )

    print(
        "\nCorrelation with Units_Sold:"
    )

    demand_correlation = (
        correlation["Units_Sold"]
        .drop("Units_Sold")
        .sort_values(
            ascending=False
        )
    )

    for feature, value in demand_correlation.items():

        print(
            f"   {feature:<35}"
            f": {value:.3f}"
        )

    # --------------------------------------------------------
    # Heatmap using matplotlib only
    # --------------------------------------------------------

    plt.figure(
        figsize=(13, 10)
    )

    plt.imshow(
        correlation,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
    )

    plt.yticks(
        range(len(correlation.index)),
        correlation.index,
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        VISUALIZATION_DIR
        / "correlation_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "📈 Saved: correlation_heatmap.png"
    )

    return correlation


# ============================================================
# AUTOMATED BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(
    df: pd.DataFrame,
    hospital_result: pd.DataFrame,
    category_result: pd.DataFrame,
    territory_result: pd.DataFrame,
    monthly_result: pd.DataFrame,
    volatility_result: pd.DataFrame,
) -> list[str]:
    """
    Generate concise business insights from the EDA results.
    """

    print(
        "\n🧠 Generating business insights..."
    )

    insights = []

    # --------------------------------------------------------
    # Top hospital
    # --------------------------------------------------------

    top_hospital = (
        hospital_result.iloc[0]
    )

    insights.append(
        "Hospital account concentration: "
        f"{top_hospital['Hospital_Name']} "
        f"is the highest-revenue account, "
        f"contributing "
        f"{top_hospital['Revenue_Share_Pct']:.2f}% "
        "of total revenue."
    )

    # --------------------------------------------------------
    # Top category
    # --------------------------------------------------------

    top_category = (
        category_result.iloc[0]
    )

    insights.append(
        "Product portfolio concentration: "
        f"{top_category['Product_Category']} "
        f"is the highest-revenue category, "
        f"representing "
        f"{top_category['Revenue_Share_Pct']:.2f}% "
        "of total revenue."
    )

    # --------------------------------------------------------
    # Top territory
    # --------------------------------------------------------

    top_territory = (
        territory_result.iloc[0]
    )

    insights.append(
        "Territory performance: "
        f"{top_territory['Territory']} "
        f"is the strongest territory by revenue, "
        f"with "
        f"₹{top_territory['Total_Revenue']:,.0f} "
        "in total revenue."
    )

    # --------------------------------------------------------
    # Best month
    # --------------------------------------------------------

    best_month = (
        monthly_result.loc[
            monthly_result["Total_Revenue"].idxmax()
        ]
    )

    insights.append(
        "Peak revenue period: "
        f"{best_month['Transaction_Date'].strftime('%Y-%m')} "
        f"generated the highest monthly revenue of "
        f"₹{best_month['Total_Revenue']:,.0f}."
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    highest_volatility = (
        volatility_result.iloc[0]
    )

    insights.append(
        "Demand volatility: "
        f"{highest_volatility['Product_Category']} "
        "shows the highest average demand volatility "
        "among product categories."
    )

    # --------------------------------------------------------
    # Stockout rate
    # --------------------------------------------------------

    stockout_rate = (
        df["Stockout_Flag"]
        .mean()
        * 100
    )

    insights.append(
        "Stockout exposure: "
        f"{stockout_rate:.2f}% "
        "of transaction observations "
        "are associated with stockout conditions."
    )

    # --------------------------------------------------------
    # Promotion rate
    # --------------------------------------------------------

    promotion_rate = (
        df["Promotion_Flag"]
        .mean()
        * 100
    )

    insights.append(
        "Promotion exposure: "
        f"{promotion_rate:.2f}% "
        "of transaction observations "
        "occurred under promotion conditions."
    )

    print(
        "\nKey business insights:"
    )

    for index, insight in enumerate(
        insights,
        start=1,
    ):

        print(
            f"\n   {index}. {insight}"
        )

    return insights


# ============================================================
# SAVE REPORT
# ============================================================

def save_eda_report(
    overview: dict,
    insights: list[str],
) -> None:
    """
    Save the EDA findings into a text report.
    """

    print(
        "\n📝 Saving EDA report..."
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "HEALTHCARE MEDICAL-DEVICE SALES\n"
        )

        file.write(
            "EDA & BUSINESS INSIGHTS REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "DATASET OVERVIEW\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(
            f"Rows: "
            f"{overview['rows']:,}\n"
        )

        file.write(
            f"Columns: "
            f"{overview['columns']:,}\n"
        )

        file.write(
            f"Hospitals: "
            f"{overview['hospitals']:,}\n"
        )

        file.write(
            f"Products: "
            f"{overview['products']:,}\n"
        )

        file.write(
            f"Product categories: "
            f"{overview['categories']:,}\n"
        )

        file.write(
            f"Territories: "
            f"{overview['territories']:,}\n"
        )

        file.write(
            f"Total units sold: "
            f"{overview['total_units']:,.0f}\n"
        )

        file.write(
            f"Total revenue: "
            f"₹{overview['total_revenue']:,.2f}\n"
        )

        file.write(
            f"Average transaction revenue: "
            f"₹{overview['average_transaction_revenue']:,.2f}\n"
        )

        file.write(
            f"Median transaction revenue: "
            f"₹{overview['median_transaction_revenue']:,.2f}\n"
        )

        file.write(
            f"Date range: "
            f"{overview['date_start'].date()} "
            f"to "
            f"{overview['date_end'].date()}\n"
        )

        file.write(
            "\n\nBUSINESS INSIGHTS\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        for index, insight in enumerate(
            insights,
            start=1,
        ):

            file.write(
                f"{index}. {insight}\n\n"
            )

    print(
        f"✅ EDA report saved to:\n"
        f"   {REPORT_FILE}"
    )


# ============================================================
# MAIN EDA PIPELINE
# ============================================================

def run_eda() -> None:
    """
    Execute the complete EDA workflow.
    """

    print("\n" + "=" * 70)

    print(
        "📊 HEALTHCARE MEDICAL-DEVICE SALES"
    )

    print(
        "EXPLORATORY DATA ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Overview
    # --------------------------------------------------------

    overview = generate_dataset_overview(
        df
    )

    # --------------------------------------------------------
    # Hospital analysis
    # --------------------------------------------------------

    hospital_result = (
        analyze_revenue_by_hospital(
            df
        )
    )

    # --------------------------------------------------------
    # Product category analysis
    # --------------------------------------------------------

    category_result = (
        analyze_product_categories(
            df
        )
    )

    # --------------------------------------------------------
    # Territory analysis
    # --------------------------------------------------------

    territory_result = (
        analyze_territories(
            df
        )
    )

    # --------------------------------------------------------
    # Monthly trends
    # --------------------------------------------------------

    monthly_result, _ = (
        analyze_monthly_trends(
            df
        )
    )

    # --------------------------------------------------------
    # Quarterly analysis
    # --------------------------------------------------------

    analyze_quarters(
        df
    )

    # --------------------------------------------------------
    # Inventory analysis
    # --------------------------------------------------------

    analyze_inventory(
        df
    )

    # --------------------------------------------------------
    # Stockout analysis
    # --------------------------------------------------------

    analyze_stockouts(
        df
    )

    # --------------------------------------------------------
    # Promotion analysis
    # --------------------------------------------------------

    analyze_promotions(
        df
    )

    # --------------------------------------------------------
    # Demand distribution
    # --------------------------------------------------------

    analyze_demand_distribution(
        df
    )

    # --------------------------------------------------------
    # Demand volatility
    # --------------------------------------------------------

    volatility_result = (
        analyze_demand_volatility(
            df
        )
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    analyze_correlations(
        df
    )

    # --------------------------------------------------------
    # Business insights
    # --------------------------------------------------------

    insights = generate_business_insights(
        df,
        hospital_result,
        category_result,
        territory_result,
        monthly_result,
        volatility_result,
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_eda_report(
        overview,
        insights,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "✅ EDA COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        "\n📁 Visualizations saved in:"
    )

    print(
        f"   {VISUALIZATION_DIR}"
    )

    print(
        "\n📄 Report saved in:"
    )

    print(
        f"   {REPORT_FILE}"
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_eda()

        sys.exit(0)

    except Exception as exc:

        print(
            "\n❌ EDA PIPELINE FAILED."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)