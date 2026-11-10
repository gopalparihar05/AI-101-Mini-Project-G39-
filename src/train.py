"""MEMBER 3 - Build and train the linear regression models.

What this file does:
1. Turn the text columns into number columns (one-hot encoding).
2. Split the data into train (80%) and test (20%). All rows of one phone stay together.
3. Train 2 models:
   - Simple linear regression: 1 feature (RAM).
   - Multiple linear regression: all features.

The models learn the log of the price. The price has a long right tail,
and the log of the price is closer to a straight-line relation.
To get a price in USD, evaluate.py changes the answer back with np.expm1.

Run only this part:  python -m src.train
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import data_prep
from src.config import (
    CATEGORICAL_FEATURES,
    GROUP_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

SIMPLE = "Simple linear regression (RAM only)"
MULTIPLE = "Multiple linear regression (all features)"


def split_data(df):
    """Return X_train, X_test, y_train, y_test. A phone model is never in both."""
    # One-hot encoding: a text column becomes one 0/1 column for each value.
    # drop_first=True removes one column of each group. It would repeat the information.
    X = pd.get_dummies(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    y = df[TARGET]

    # Choose 20% of the phone models for the test data. Their rows all go to the test data.
    test_phones = df[GROUP_COLUMN].drop_duplicates().sample(
        frac=TEST_SIZE, random_state=RANDOM_STATE
    )
    is_test = df[GROUP_COLUMN].isin(test_phones)
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


def train_models(X_train, y_train):
    """Train both models. Return a dict: name -> (model, list of columns the model uses)."""
    log_price = np.log1p(y_train)
    models = {}
    for name, columns in [(SIMPLE, ["ram"]), (MULTIPLE, list(X_train.columns))]:
        model = LinearRegression()
        model.fit(X_train[columns], log_price)
        models[name] = (model, columns)
    return models


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = split_data(data_prep.load_clean())
    print(f"Train rows: {len(X_tr)}  Test rows: {len(X_te)}  Columns used: {X_tr.shape[1]}")
    for model_name, (fitted, used_columns) in train_models(X_tr, y_tr).items():
        score = fitted.score(X_tr[used_columns], np.log1p(y_tr))
        print(f"{model_name}: R2 on the train data (log price) = {score:.3f}")
