"""Run the full project: clean -> explore -> train -> evaluate.

Command (from the project folder):  python run_all.py
All results go to the outputs/ folder.
"""
from src import data_prep, eda, evaluate, train


def main():
    print("Step 1/4 (Member 1): load and clean the data")
    df = data_prep.load_clean()
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    print("Step 2/4 (Member 2): exploratory data analysis")
    for file in eda.run(df):
        print("  saved", file.name)

    print("Step 3/4 (Member 3): train the linear regression models")
    X_train, X_test, y_train, y_test = train.split_data(df)
    models = train.train_models(X_train, y_train)
    print(f"  train rows: {len(X_train)}, test rows: {len(X_test)}")

    print("Step 4/4 (Member 4): evaluate on the test data")
    table, charts = evaluate.run(models, X_test, y_test)
    print(table.to_string())
    for chart in charts:
        print("  saved", chart.name)

    print("\nDone. See the outputs/ folder.")


if __name__ == "__main__":
    main()
