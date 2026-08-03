# How Data Quality Affects Energy Demand Forecasting in UK Smart Grids

**Author:** Bhavin Sanjaybhai Sangani (P2957969)
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
│   ├── Step1_DataCollection_EDA.ipynb        # Data download, cleaning, EDA
│   ├── Step2_Validation_Tuning.ipynb         # Walk-forward validation + hyperparameter tuning
│   └── Step3_DefectInjection_Imputation.ipynb # (in progress) defect injection + imputation
├── output/
│   ├── clean_demand_2024_2025.csv            # Verified clean baseline dataset
│   └── best_hyperparameters.json             # Tuned model settings
└── README.md
```

## Progress

| Step | Status | Description |
|---|---|---|
| Step 1 — Data Collection & EDA | ✅ Complete | Downloaded, merged, cleaned, and explored 2 years of NESO demand data |
| Step 2 — Validation & Tuning | ✅ Complete | Walk-forward validation implemented; ARIMA, Linear Regression, Random Forest and LSTM tuned/evaluated |
| Step 3 — Defect Injection & Imputation | ⏳ In progress | Introducing 4 defect types and testing 4 imputation methods |
| Modelling on degraded data | ⏳ Not started | Running all 4 models across all defect/imputation combinations |
| Results analysis & write-up | ⏳ Not started | — |

## Step 1: Data Collection & EDA

- Downloaded Historic Demand Data (2024, 2025) directly from the NESO Data Portal API — 35,088 half-hourly readings.
- Built a continuous datetime index. An initial version calculated timestamps from the settlement period number, which broke on the UK's two clock-change days per year (50-period October day, 46-period March day), producing 4 false duplicate timestamps. Fixed by assigning timestamps purely sequentially, since the source data is already in chronological order.
- Verified: 0 duplicate timestamps, 0 missing values, 35,088 rows retained.
- Cross-checked descriptive statistics against NESO's own published 2024 figures (max 45,924 MW here vs NESO's reported 45,096 MW) as an independent validation.
- Produced EDA plots confirming daily and weekly seasonality, consistent with Taylor (2003).

## Step 2: Validation & Hyperparameter Tuning

- Implemented walk-forward (rolling-origin) validation, replacing standard k-fold, so no model is ever trained on data from after the point it's being tested on.
- Tuned Random Forest (9 combinations of `n_estimators` × `max_depth`) and LSTM (4 combinations of `units` × `dropout`), each scored using the same walk-forward folds.
- Found and corrected an unfair test setup for ARIMA, which was originally forecasting several months ahead in a single pass rather than one step at a time like the other models — this had inflated its error by roughly 18×.

### Results on clean baseline data

| Model | RMSE (MW) |
|---|---|
| Linear Regression | 389.80 |
| Random Forest (n_estimators=500, max_depth=30) | 402.82 |
| LSTM (units=64, dropout=0.1) | 445.88 |
| ARIMA (2,1,2) | 486.52 |

These are the baseline figures every degraded/imputed dataset in Step 3 will be compared against.

## How to Run

1. Open a notebook in [Google Colab](https://colab.research.google.com).
2. Run cells in order from top to bottom.
3. Step 2 and Step 3 notebooks require `output/clean_demand_2024_2025.csv` from Step 1 to already exist in the session.
4. For LSTM tuning (Step 2), enable a GPU runtime: `Runtime > Change runtime type > GPU`.

## Methodology Notes

- **Evaluation metric:** RMSE, prioritised over MAE since large forecast errors carry disproportionate real-world cost in grid balancing.
- **Validation:** Walk-forward (rolling-origin), not k-fold, to preserve temporal order.
- **Consistency:** Once tuned, each model's hyperparameters remain fixed across every degraded/imputed dataset in Step 3, so accuracy differences reflect data quality rather than inconsistent tuning.

## References

Full literature review and citations are maintained in the project proposal and supervisor-facing documentation (not included in this repository).
