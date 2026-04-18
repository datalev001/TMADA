# Maintaining Forecast Accuracy in Deployed Time-Series Models
# Keeping Forecasts Alive in Production

This repository contains the code and experiment outputs for a practical study on how to keep a deployed time-series forecasting model useful after it starts to age.

The project compares five update strategies on the monthly Electric Production dataset:

1. Adaptive base SARIMA
2. Adaptive correction + refresh
3. Frozen correction + gate
4. Kalman bias-corrected
5. Frozen base SARIMA

The goal is simple: when new observations arrive, should we keep the old model, correct it, track its bias, or refresh it?

## Why this project matters

A forecasting model often performs well at deployment, then gradually becomes stale.

In real business settings, a team usually does not want to retrain the whole forecasting system every time one new observation arrives. That can be expensive, unstable, and operationally inconvenient. At the same time, leaving a model untouched for too long can also be costly.

This repository explores several middle-ground strategies for short-horizon forecasting maintenance.

## Dataset

The experiments use the monthly Electric Production time series.

- Source: Kaggle Electric Production dataset
- Frequency: Monthly
- Number of observations: 397
- Main target column: `IPG2211A2N`

The original experiments used the following split:

- Train: 1985-01 to 2006-12
- Calibration: 2007-01 to 2013-12
- Test: 2014-01 to 2018-01

All reported results focus on one-step-ahead forecasting (H = 1).

## Methods included

### 1. Frozen base SARIMA
Train a SARIMA model once and keep using it without refitting.

### 2. Frozen correction + gate
Keep the base model fixed, then add a small correction layer that predicts when the base forecast is likely to be wrong and whether the correction should be applied.

### 3. Kalman bias-corrected
Treat the forecast bias as a slowly changing hidden state and update that bias recursively.

### 4. Adaptive base SARIMA
Allow the SARIMA backbone itself to be refreshed when recent performance suggests that the model has become too old.

### 5. Adaptive correction + refresh
Use correction first, but allow the system to refit the base model when correction is no longer enough.


## Main result table

| Method | MAE | RMSE | sMAPE | MASE |
|---|---:|---:|---:|---:|
| Adaptive base SARIMA | 2.6812 | 3.4214 | 2.5519 | 0.9969 |
| Adaptive correction + refresh | 2.7222 | 3.5357 | 2.5807 | 1.0121 |
| Frozen correction + gate | 2.7229 | 3.4983 | 2.5862 | 1.0124 |
| Kalman bias-corrected | 2.7538 | 3.4972 | 2.6290 | 1.0239 |
| Frozen base SARIMA | 2.7659 | 3.4941 | 2.6426 | 1.0284 |


## Key takeaway

The main lesson from this project is straightforward:

Keeping the base model from becoming too old matters more than building an overly complicated correction layer.

The frozen correction approach does help. The Kalman-style bias tracker also helps a little. But on this dataset, the strongest gains came from allowing the base model itself to refresh when needed.

## Files in this repository

- `five_methods_experiment.py`  
  Full experiment script for the five-method comparison.

- `five_methods_results.csv`  
  Final comparison table used in the article.

- `Electric_Production.csv`  
  Dataset file used in the experiments.  
  If you use a different filename or path, update `DATA_PATH` in the script.

---

## Requirements

Recommended Python version:
- Python 3.10 or 3.11

Core packages:
- `numpy`
- `pandas`
- `scikit-learn`
- `statsmodels`

You can install them with:

```bash
pip install numpy pandas scikit-learn statsmodels
```

## How to run

1. Put the dataset CSV in the project folder.
2. Update `DATA_PATH` in `five_methods_experiment.py` if needed.
3. Run:

```bash
python five_methods_experiment.py
```

The script will print the comparison table and save:

```text
outputs/five_methods_results.csv
```

## Reproducibility note

The experiment script is written to be easy to read and modify.  
Several thresholds and hyperparameters are intentionally exposed near the top of the script so that you can test different operational assumptions.

These include:
- correction gate threshold
- clipping scale
- adaptive refresh trigger
- rolling window length
- Kalman tuning grid

## Suggested use

This repository may be useful if you are working on:

- demand forecasting
- inventory planning
- utility load monitoring
- operational short-horizon forecasting
- model maintenance after deployment

It is especially relevant when retraining every time is unrealistic, but leaving the model untouched is also unsatisfactory.

## Citation / article note

This code accompanies an article discussing practical ways to maintain deployed forecasting models after they begin to age.

See code and data in my GitHub repository for full reproducibility.

