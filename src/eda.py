"""MEMBER 2 - Exploratory data analysis (EDA) and charts.

What this file does:
1. Write a text summary of the clean data.
2. Save 4 charts to outputs/figures/. Use them on the PPT slides.

Run only this part:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")  # Save charts to files. No window opens.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import data_prep  # noqa: E402
from src.config import FIGURE_DIR, NUMERIC_FEATURES, OUTPUT_DIR, TARGET  # noqa: E402


def save_chart(name):
    """Save the current chart to outputs/figures/ and close it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def write_summary(df):
    """Write the size of the data and the statistics of the numeric columns to a text file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "eda_summary.txt"
    text = (
        f"Rows: {len(df)}\n"
        f"Columns: {df.shape[1]}\n"
        f"Different phone models: {df['phone_model'].nunique()}\n\n"
        f"Statistics of the numeric columns:\n{df.describe().round(2).T.to_string()}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def plot_price_distribution(df):
    """The price has a long right tail. The log of the price is closer to a bell shape."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df[TARGET], bins=50)
    plt.title("Price (USD)")
    plt.xlabel("price_usd")
    plt.ylabel("Number of rows")
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df[TARGET]), bins=50)
    plt.title("Log of the price")
    plt.xlabel("log(1 + price_usd)")
    return save_chart("01_price_distribution.png")


def plot_correlation(df):
    """Correlation of each numeric feature with the log price."""
    correlation = df[NUMERIC_FEATURES].corrwith(np.log1p(df[TARGET])).sort_values()
    plt.figure(figsize=(7, 5))
    plt.barh(correlation.index, correlation.values)
    plt.xlabel("Correlation with log price")
    plt.title("Which numeric columns go with the price?")
    return save_chart("02_correlation.png")


def plot_price_by_brand(df):
    """Median price of the 10 brands that have the most rows."""
    top_brands = df["phone_brand"].value_counts().head(10).index
    medians = df[df["phone_brand"].isin(top_brands)].groupby("phone_brand")[TARGET].median()
    medians = medians.sort_values()
    plt.figure(figsize=(7, 5))
    plt.barh(medians.index, medians.values)
    plt.xlabel("Median price (USD)")
    plt.title("Median price by brand (10 brands with most rows)")
    return save_chart("03_price_by_brand.png")


def plot_price_vs_ram(df):
    """Price against RAM. More RAM usually means a higher price."""
    plt.figure(figsize=(7, 5))
    plt.scatter(df["ram"], df[TARGET], alpha=0.3, s=12)
    plt.xlabel("RAM (GB)")
    plt.ylabel("Price (USD)")
    plt.title("Price against RAM")
    return save_chart("04_price_vs_ram.png")


def run(df):
    """Make the summary file and all charts. Return the list of files."""
    return [
        write_summary(df),
        plot_price_distribution(df),
        plot_correlation(df),
        plot_price_by_brand(df),
        plot_price_vs_ram(df),
    ]


if __name__ == "__main__":
    for file in run(data_prep.load_clean()):
        print("Saved", file)
