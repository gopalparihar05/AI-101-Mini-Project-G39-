"""MEMBER 1 - Data loading and cleaning.

What this file does:
1. Load the CSV file.
2. Remove the columns that leak the price (see config.LEAKAGE_COLUMNS).
3. Make the text columns lower case, so 'Mediatek' and 'MediaTek' are the same.
4. Make one new column: launch_year.
5. Keep only the columns we need.

Run only this part:  python -m src.data_prep
"""
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    GROUP_COLUMN,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET,
)


def load_raw(path=DATA_PATH):
    """Read the CSV file into a DataFrame."""
    return pd.read_csv(path)


def clean(raw):
    """Return a clean DataFrame with the price, the phone name and all features."""
    # 1. Remove the columns that are made from the price.
    df = raw.drop(columns=LEAKAGE_COLUMNS)

    # 2. Text columns: lower case and no extra spaces.
    for column in CATEGORICAL_FEATURES + [GROUP_COLUMN]:
        df[column] = df[column].str.strip().str.lower()

    # 3. launch_date looks like '2024-09-20'. The first 4 characters are the year.
    df["launch_year"] = df["launch_date"].str[:4].astype(int)

    # 4. Keep the columns we need. Remove rows that have a missing value.
    keep = [TARGET, GROUP_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df[keep].dropna()
    return df.reset_index(drop=True)


def load_clean():
    """Load the CSV file and clean it. The other parts of the project call this function."""
    return clean(load_raw())


if __name__ == "__main__":
    data = load_clean()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_DIR / "clean_data.csv", index=False)
    print(f"Clean data: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Saved to {OUTPUT_DIR / 'clean_data.csv'}")
