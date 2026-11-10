"""Shared settings. All four members import from this file.

Change a value here, and all four parts of the project use the new value.
"""
from pathlib import Path

# Paths are relative to the project folder, so the code runs on any laptop.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_data2.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

TARGET = "price_usd"
# The same phone appears in many rows (different store or storage size).
# We put all rows of one phone on the same side of the train/test split.
GROUP_COLUMN = "phone_model"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# These columns are made from the price. If we use them, the model sees the answer.
LEAKAGE_COLUMNS = ["quantile_10", "quantile_50", "quantile_90", "price_range"]

NUMERIC_FEATURES = [
    "storage",
    "ram",
    "weight",
    "display_size",
    "battery",
    "ppi_density",
    "colors_available",
    "nfc",
    "foldable",
    "launch_year",  # made in data_prep.py from launch_date
]

CATEGORICAL_FEATURES = [
    "phone_brand",
    "store",
    "os_type",
    "chip_company",
]
