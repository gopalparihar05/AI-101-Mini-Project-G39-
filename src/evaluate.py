"""MEMBER 4 - Evaluate the models on the test data and make the results charts.

What this file does:
1. Score both models on the test data (data the model never saw).
2. Save 2 charts: actual against predicted, and residuals.

Metrics:
- R2: how much of the price change the model explains. 1.0 is perfect, 0.0 is no better than the mean.
- RMSE and MAE: the average error in USD.

Run only this part:  python -m src.evaluate
"""
import matplotlib

matplotlib.use("Agg")  # Save charts to files. No window opens.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from src import data_prep, train  # noqa: E402
from src.config import FIGURE_DIR, OUTPUT_DIR  # noqa: E402


def predict_price(model, columns, X):
    """The model predicts the log price. np.expm1 changes it back to USD."""
    return np.expm1(model.predict(X[columns]))


def evaluate_models(models, X_test, y_test):
    """Score all models on the test data. Save and return the table."""
    rows = {}
    for name, (model, columns) in models.items():
        predicted = predict_price(model, columns, X_test)
        rows[name] = {
            "R2": r2_score(y_test, predicted),
            "RMSE_usd": np.sqrt(mean_squared_error(y_test, predicted)),
            "MAE_usd": mean_absolute_error(y_test, predicted),
        }
    table = pd.DataFrame(rows).T.round(3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "metrics_test.csv")
    return table


def save_chart(name):
    """Save the current chart to outputs/figures/ and close it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_actual_vs_predicted(predicted, y_test):
    """Each dot is one row. A dot on the red line is a perfect prediction."""
    top = max(y_test.max(), predicted.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predicted, alpha=0.4, s=14)
    plt.plot([0, top], [0, top], color="red")
    plt.xlabel("Actual price (USD)")
    plt.ylabel("Predicted price (USD)")
    plt.title("Test data: actual against predicted")
    return save_chart("05_actual_vs_predicted.png")


def plot_residuals(predicted, y_test):
    """Residual = actual price - predicted price. Good residuals are around 0."""
    plt.figure(figsize=(7, 4.5))
    plt.scatter(predicted, y_test - predicted, alpha=0.4, s=14)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted price (USD)")
    plt.ylabel("Residual (USD)")
    plt.title("Test data: residuals")
    return save_chart("06_residuals.png")


def run(models, X_test, y_test):
    """Make the metrics table and the 2 result charts."""
    table = evaluate_models(models, X_test, y_test)
    model, columns = models[train.MULTIPLE]
    predicted = predict_price(model, columns, X_test)
    files = [
        plot_actual_vs_predicted(predicted, y_test),
        plot_residuals(predicted, y_test),
    ]
    return table, files


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = train.split_data(data_prep.load_clean())
    result, charts = run(train.train_models(X_tr, y_tr), X_te, y_te)
    print("Test data results:")
    print(result.to_string())
    for chart in charts:
        print("Saved", chart)
