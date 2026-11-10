# Smartphone Price Prediction with Linear Regression

AI101 Mini-Project, IIT Ropar. We predict the price (USD) of a smartphone from its specifications. We use linear regression.

## Team and roles

Team number: `<TEAM NUMBER>`

| Member    | GitHub handle       | Code part | Lead for |
|-----------|---------------------|-----------|----------|
| `Nikhil`  | `nikh0024inst`      | `src/data_prep.py` (load and clean the data) | Documentation (README, project structure) |
| `Rishu`   | `Rajexp1559`        | `src/eda.py` (exploration and charts) | PPT (5 slides) |
| `Pranjal` | `Pranjal263`        | `src/train.py`, `run_all.py`, `requirements.txt` (models) | Setup and steps to execute |
| `Gopal`   | `gopalparihar05`    | `src/evaluate.py` (test results and charts) | Video (5 minutes) |

## Project structure

```
.
├── README.md
├── requirements.txt        Pinned library versions
├── run_all.py              Runs the full project
├── data/
│   └── processed_data2.csv Dataset (1708 rows, 38 columns)
├── src/
│   ├── config.py           Shared settings (paths, column lists)
│   ├── data_prep.py        Member 1
│   ├── eda.py              Member 2
│   ├── train.py            Member 3
│   └── evaluate.py         Member 4
└── outputs/                Made by the code (tables and charts)
```

## Dataset

- Source: Kaggle, smartphone prices from online stores (file `processed_data2.csv`). `<ADD THE KAGGLE LINK>`
- Each row is one phone in one store with one storage size. The target is `price_usd`.
- We remove 4 columns (`quantile_10`, `quantile_50`, `quantile_90`, `price_range`). They are made from the price, so they give the answer to the model.
- A phone model appears in many rows. We keep all rows of one phone model on the same side of the train/test split. Without this, the test score looks better than it is.

## Setup

Python 3.9 to 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Steps to run

1. Open a terminal in the project folder.
2. Run the full project:
   ```bash
   python run_all.py
   ```
3. Open the `outputs/` folder to see the tables and charts.

Each member can also run one part:

```bash
python -m src.data_prep   # Member 1: clean data
python -m src.eda         # Member 2: charts
python -m src.train       # Member 3: train the models
python -m src.evaluate    # Member 4: test results
```

## Method

1. Clean the data (`data_prep.py`). Remove the 4 leakage columns. Make 1 new column: `launch_year`.
2. Turn the text columns into 0/1 columns with `pd.get_dummies` (one-hot encoding).
3. Split the data: 80% train, 20% test. All rows of one phone model go to the same side.
4. Fit linear regression on `log(1 + price)`. The price has a long right tail. The code changes the answer back to USD.
5. Train 2 models: simple linear regression (RAM only) and multiple linear regression (all features).
6. Score both models on the test data: R2, RMSE, MAE.

## Expected output

`<ADD A SCREENSHOT OF THE TERMINAL AFTER python run_all.py>`

Files in `outputs/`: `clean_data.csv`, `eda_summary.txt`, `metrics_test.csv`, and 6 charts in `outputs/figures/`.

## Results

`<ADD THE TABLE FROM outputs/metrics_test.csv AFTER YOU RUN THE CODE>`

## Links

- Slides: `<LINK OR FILE PATH>`
- Video: `<LINK>`

## AI-use disclosure

- Claude (Anthropic): wrote the first draft of the code in `src/`, `run_all.py` and this README. `<NAME>` checked the code, ran it on a fresh clone, and changed `<WHAT YOU CHANGED>`.
- `<Add every other AI tool you used: tool, what for, how you checked it.>`
- No AI was used to make the dataset or the results.
