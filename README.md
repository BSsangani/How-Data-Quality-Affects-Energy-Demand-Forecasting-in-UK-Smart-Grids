# How Data Quality Affects Energy Demand Forecasting in UK Smart Grids

**Author:** Bhavin Sanjaybhai Sangani
**Module:** CSIP5402 – Project Proposal, Planning & Project Management

## Overview

This project investigates how different types of data quality defects — missing values, sensor blackouts, noise, and timestamp errors — affect the accuracy of electricity demand forecasting models. Using two years of real half-hourly demand data from Great Britain (NESO), the project deliberately introduces controlled defects into a clean baseline dataset, tests four forecasting models (ARIMA, Linear Regression, Random Forest, LSTM) against both clean and degraded versions, and evaluates four imputation strategies for recovering lost accuracy.

Most existing research either works with pre-cleaned datasets or tests synthetic data rather than real grid data. This project addresses that gap by combining real NESO data, progressive multi-type degradation, and a direct comparison between statistical and machine learning forecasting approaches.

## Research Questions

- **RQ1** — Does missing data (random or block gaps) affect ARIMA vs ML forecast accuracy differently, and does the effect scale with the percentage of missing data?
- **RQ2** — Does Gaussian noise or timestamp irregularity affect statistical vs ML approaches differently, and are ML models more robust to these defects?
- **RQ3** — Which imputation strategy best restores accuracy across all four models, and under what conditions does it fail?
- **RQ4** — Is there a degradation threshold beyond which no model/imputation combination can recover accuracy?
- **RQ5** — Are Random Forest and LSTM more resilient to degraded data than ARIMA?

## Data Source

- **Source:** [NESO Data Portal](https://www.neso.energy/data-portal) — Historic Demand Data
- **Period:** 1 January 2024 – 31 December 2025 (two full calendar years)
- **Resolution:** Half-hourly (48 readings per day)
- **Target variable:** `ND` (National Demand, MW)

## Repository Structure

```
├── notebooks/
│   ├── Step1_DataCollection_EDA.ipynb          # Data download, cleaning, EDA
│   ├── Step2_Validation_Tuning.ipynb           # Walk-forward validation + hyperparameter tuning
│   ├── Step3_DefectInjection_Imputation.ipynb  # Defect injection + imputation
│   └── Step4_ModellingOnDegradedData.ipynb     # Running all 4 models on every scenario
├── output/
│   ├── clean_demand_2024_2025.csv              # Verified clean baseline dataset
│   ├── best_hyperparameters.json               # Tuned model settings from Step 2
│   ├── degraded/                               # 6 damaged copies of the clean data
│   ├── imputed/                                # 16 imputed (repaired) datasets
│   ├── imputation_recovery_summary.csv         # How well each fix recovered the true values
│   └── step4_model_comparison.csv              # RMSE of all 4 models on every scenario
└── README.md
```

## Progress

| Step | Status | Description |
|---|---|---|
| Step 1 — Data Collection & EDA | ✅ Complete | Downloaded, merged, cleaned, and explored 2 years of NESO demand data |
| Step 2 — Validation & Tuning | ✅ Complete | Walk-forward validation implemented; ARIMA, Linear Regression, Random Forest and LSTM tuned/evaluated |
| Step 3 — Defect Injection & Imputation | ✅ Complete | Introduced 4 defect types, tested 4 imputation methods, measured recovery |
| Step 4 — Modelling on Degraded Data | ⏳ In progress | Running all 4 tuned models on the clean baseline plus 18 degraded/imputed scenarios |
| Results analysis & write-up | ⏳ Not started | — |

## How to Run This Project

All notebooks are designed for **Google Colab**, and read/write everything from a single Google Drive folder so nothing needs to be re-uploaded between sessions.

### One-time setup

1. In Google Drive, create a folder called `energy-project`, and inside it a folder called `output`.
2. Open each notebook in [Google Colab](https://colab.research.google.com) (`File > Upload notebook`, or open directly from GitHub).
3. The first code cell in every notebook connects to Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   os.chdir('/content/drive/MyDrive/energy-project')
   ```
   Run it and approve the Google account permission prompt when asked. Every `output/...` path used later in the notebook will then automatically read from and write to this Drive folder — no manual downloading or uploading required between sessions.

### Running order

Run the notebooks in this order, top to bottom, since each one depends on files produced by the last:

1. **Step 1 — Data Collection & EDA.** Downloads the NESO data directly from its public URLs and builds `output/clean_demand_2024_2025.csv`. Takes a few minutes.
2. **Step 2 — Validation & Tuning.** Needs Step 1's clean CSV. Tunes Random Forest and LSTM, evaluates ARIMA and Linear Regression, and saves `output/best_hyperparameters.json`. Enable a GPU runtime (`Runtime > Change runtime type > T4 GPU`) before the LSTM section — Random Forest tuning alone can take 15–20 minutes on CPU regardless of GPU, since scikit-learn doesn't use it.
3. **Step 3 — Defect Injection & Imputation.** Needs Step 1's clean CSV. Creates 6 damaged datasets and 16 imputed datasets under `output/degraded/` and `output/imputed/`. Runs in a few minutes; no GPU needed.
4. **Step 4 — Modelling on Degraded Data.** Needs Step 2's `best_hyperparameters.json` and Step 3's degraded/imputed files. Runs all 4 models against 18 scenarios. This is the slowest notebook — ARIMA's one-step-ahead evaluation alone takes roughly 2–3 minutes per scenario, so the full run can take over an hour. Results are saved after every single scenario, and the notebook automatically skips anything already completed, so if the session disconnects partway through, just reconnect and re-run the same cell — it resumes rather than starting over.

### Keeping GitHub in sync

Google Colab does not automatically sync with GitHub. After making any change in a notebook, update GitHub manually: `File > Save a copy in GitHub`, select the same repository and the same file path as before (so it updates the existing file rather than creating a duplicate), write a short commit message, and confirm.

## Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Reading, merging, and manipulating the half-hourly demand tables throughout every step |
| `numpy` | Array operations — generating random defect positions, computing RMSE, building lag-feature matrices |
| `requests` | Downloading the raw CSV files directly from the NESO Data Portal (Step 1) |
| `matplotlib` | All exploratory plots — the two-week window, hourly/daily averages, autocorrelation (Step 1) |
| `statsmodels` | The ARIMA baseline model, and the Kalman filter (via a `SARIMAX` state-space model) used in imputation |
| `scikit-learn` | `RandomForestRegressor`, `LinearRegression`, and `mean_squared_error` for scoring every model |
| `tensorflow` / `keras` | Building and training the LSTM model (`Sequential`, `LSTM`, `Dropout`, `Dense` layers) |
| `itertools.product` | Generating every combination of hyperparameters to search over during tuning (Step 2) |
| `json` | Saving and loading the tuned hyperparameter settings between notebooks |

## The Four Models

**ARIMA (AutoRegressive Integrated Moving Average)** — the classic statistical baseline. Configured as ARIMA(2,1,2): it predicts the next reading from the last 2 actual values, after removing the overall trend once (differencing), and also corrects itself using its last 2 prediction errors. Evaluated one step at a time, feeding in the real observed value before each new prediction, so its test is directly comparable to the other three models rather than forecasting blind for months at a stretch.

**Linear Regression** — the simplest model in the comparison, with no hyperparameters to tune. Given the previous 48 half-hourly readings (the last 24 hours) as input, it fits a straight-line relationship to predict the next reading.

**Random Forest** — an ensemble of many decision trees (tuned to 500 trees, max depth 30), each trained on a slightly different view of the same 48-reading input window, with the final prediction being the average across all of them.

**LSTM (Long Short-Term Memory)** — a recurrent neural network (tuned to 64 units, 0.1 dropout) designed to learn patterns across a sequence of readings, given the same 48-reading input window as Random Forest and Linear Regression.

All four models are evaluated using **walk-forward (rolling-origin) validation** rather than standard k-fold cross-validation, so a model is never trained on data from after the point it's being tested on — training data always precedes test data in time, matching how a real forecaster would actually operate.

## Methodology Notes

- **Evaluation metric:** RMSE, prioritised over MAE since large forecast errors carry disproportionate real-world cost in grid balancing.
- **Validation:** Walk-forward (rolling-origin), not k-fold, to preserve temporal order.
- **Consistency:** Once tuned in Step 2, each model's hyperparameters remain fixed across every degraded/imputed dataset in Steps 3–4, so accuracy differences reflect data quality rather than inconsistent tuning.

## Key Findings So Far

- An initial timestamp-construction bug (assuming every day has exactly 48 half-hour periods) collided on the UK's clock-change days, which have 46 or 50 periods — fixed by assigning timestamps purely sequentially instead of calculating them from the settlement period number.
- ARIMA's first evaluation was 18× worse than the other models due to being tested on a much harder multi-month blind forecast rather than a fair one-step-ahead comparison — corrected in Step 2.
- On clean data, the simplest models won: Linear Regression (RMSE 389.80) and Random Forest (402.82) both beat LSTM (445.88) and ARIMA (486.52).
- Interpolation was the clear best imputation method for scattered missing values, but every method — including the Kalman filter — broke down on a sustained 24–48 hour block gap, a strong early candidate for answering RQ4.

## References

Full literature review and citations are maintained in the project proposal and supervisor-facing documentation (not included in this repository).
