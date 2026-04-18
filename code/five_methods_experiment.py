
import math
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
# Replace DATA_PATH with your local path to the Kaggle
# Electric Production CSV if needed.
DATA_PATH = "Electric_Production.csv"
DATE_COL = "DATE"
TARGET_COL = "IPG2211A2N"

TRAIN_END = "2006-12-01"
CAL_END = "2013-12-01"

# Base SARIMA specification used throughout the article
BASE_ORDER = (1, 1, 1)
BASE_SEASONAL_ORDER = (0, 1, 1, 12)
BASE_TREND = "c"

SEASONAL_PERIOD = 12
LOCAL_WINDOW = 36

# Frozen correction + gate settings
RIDGE_ALPHA = 1.0
LOGIT_C = 1.0
MIN_OOF_TRAIN = 24
DELTA_SCALE = 0.02
GATE_THRESHOLD = 0.50
CLIP_KAPPA = 1.0

# Adaptive refresh settings
TRIGGER_MULT = 1.35
ROLL_WINDOW = 6
MAX_AGE = 24

# Kalman bias-tracker tuning grid
Q_GRID = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
R_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# ============================================================
# HELPERS
# ============================================================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom)))

def mase(y_true, y_pred, denom):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom)

def metrics_row(name, y_true, y_pred, mase_denom):
    return {
        "Method": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, mase_denom),
    }

def seasonal_naive(history, season=12):
    history = np.asarray(history, dtype=float)
    if len(history) >= season:
        return float(history[-season])
    return float(history[-1])

def local_ets_forecast(history, window=36, season=12):
    hist = np.asarray(history[-min(window, len(history)):], dtype=float)
    if len(hist) < 2 * season:
        return seasonal_naive(history, season)
    try:
        fit = ExponentialSmoothing(
            hist,
            trend="add",
            seasonal="add",
            seasonal_periods=season,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)
        return float(np.asarray(fit.forecast(1))[0])
    except Exception:
        return seasonal_naive(history, season)

def fit_filter(history, order, seasonal_order, trend, maxiter=30):
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    params = model.fit(disp=False, maxiter=maxiter, return_params=True)
    return model.filter(params, cov_type="none"), params

def make_frozen_forecaster(order, seasonal_order, trend, params):
    def forecast_one(history):
        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return float(np.asarray(model.filter(params, cov_type="none").forecast(1))[0])
    return forecast_one

def build_feature_row(dates, history, anchor_idx, b_pred, s_pred, l_pred, resid_map):
    def get_recent(k):
        vals = []
        j = anchor_idx - 1
        while len(vals) < k:
            vals.append(float(resid_map.get(j, 0.0)))
            j -= 1
            if j < 0:
                break
        if len(vals) < k:
            vals.extend([0.0] * (k - len(vals)))
        return vals

    r6 = get_recent(6)
    r12 = get_recent(12)
    r1, r2, r3, _, _, _ = r6

    slope6 = 0.0
    cp = 0.0
    if len(history) >= 6:
        yy = history[-6:]
        slope6 = float(np.polyfit(np.arange(len(yy)), yy, 1)[0])

    if len(history) >= 12:
        recent = history[-6:]
        prev = history[-12:-6]
        cp = float(abs(np.mean(recent) - np.mean(prev)) / (np.std(history[-12:]) + 1e-6))

    m = int(dates.iloc[anchor_idx].month)
    return {
        "base_forecast": b_pred,
        "probe_seasonal_naive": s_pred,
        "probe_local_ets": l_pred,
        "disagree_seasonal_naive": s_pred - b_pred,
        "disagree_local_ets": l_pred - b_pred,
        "r_t1": r1,
        "r_t2": r2,
        "r_t3": r3,
        "r_mean_3": float(np.mean([r1, r2, r3])),
        "r_mean_6": float(np.mean(r6)),
        "r_mae_6": float(np.mean(np.abs(r6))),
        "r_std_6": float(np.std(r6)),
        "r_std_12": float(np.std(r12)),
        "slope_6": slope6,
        "changepoint_6_6": cp,
        "sin_month": math.sin(2 * math.pi * m / 12.0),
        "cos_month": math.cos(2 * math.pi * m / 12.0),
    }

def load_data(data_path):
    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    dates = df[DATE_COL]
    y = df[TARGET_COL].astype(float).values

    train_end_idx = int(df.index[df[DATE_COL] <= pd.Timestamp(TRAIN_END)][-1])
    cal_end_idx = int(df.index[df[DATE_COL] <= pd.Timestamp(CAL_END)][-1])
    cal_start_idx = train_end_idx + 1
    test_start_idx = cal_end_idx + 1

    mase_denom = float(np.mean(np.abs(
        y[SEASONAL_PERIOD:train_end_idx + 1] -
        y[:train_end_idx + 1 - SEASONAL_PERIOD]
    )))

    return {
        "df": df,
        "dates": dates,
        "y": y,
        "train_end_idx": train_end_idx,
        "cal_end_idx": cal_end_idx,
        "cal_start_idx": cal_start_idx,
        "test_start_idx": test_start_idx,
        "mase_denom": mase_denom,
    }

# ============================================================
# METHOD 1 + 2: FROZEN BASE SARIMA + FROZEN CORRECTION/GATE
# ============================================================
def run_frozen_methods(ctx):
    dates = ctx["dates"]
    y = ctx["y"]
    train_end_idx = ctx["train_end_idx"]
    cal_end_idx = ctx["cal_end_idx"]
    cal_start_idx = ctx["cal_start_idx"]
    test_start_idx = ctx["test_start_idx"]
    mase_denom = ctx["mase_denom"]

    train_y = y[:train_end_idx + 1]
    base_fit = SARIMAX(
        train_y,
        order=BASE_ORDER,
        seasonal_order=BASE_SEASONAL_ORDER,
        trend=BASE_TREND,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=100)

    base_forecast = make_frozen_forecaster(
        BASE_ORDER, BASE_SEASONAL_ORDER, BASE_TREND, base_fit.params
    )

    train_resid = np.asarray(base_fit.resid, dtype=float)
    resid_map = {i: float(train_resid[i]) for i in range(len(train_resid))}

    rows = []
    for i in range(cal_start_idx, len(y)):
        history = y[:i]
        b = base_forecast(history)
        s = seasonal_naive(history, SEASONAL_PERIOD)
        l = local_ets_forecast(history, LOCAL_WINDOW, SEASONAL_PERIOD)
        feat = build_feature_row(dates, history, i, b, s, l, resid_map)
        err = float(y[i] - b)

        row = dict(feat)
        row.update({
            "idx": i,
            "date": dates.iloc[i],
            "actual": y[i],
            "base_pred": b,
            "seasonal_naive_pred": s,
            "local_ets_pred": l,
            "base_error": err,
        })
        rows.append(row)
        resid_map[i] = err

    roll_df = pd.DataFrame(rows)
    cal_df = roll_df[roll_df["idx"] <= cal_end_idx].reset_index(drop=True)
    test_df = roll_df[roll_df["idx"] >= test_start_idx].reset_index(drop=True)

    feature_cols = [
        c for c in cal_df.columns
        if c not in ["idx", "date", "actual", "base_pred", "seasonal_naive_pred", "local_ets_pred", "base_error"]
    ]

    X_cal = cal_df[feature_cols].values
    y_cal = cal_df["base_error"].values

    oof = np.full(len(cal_df), np.nan)
    for i in range(MIN_OOF_TRAIN, len(cal_df)):
        mdl = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ])
        mdl.fit(X_cal[:i], y_cal[:i])
        oof[i] = mdl.predict(X_cal[i:i+1])[0]

    delta = DELTA_SCALE * float(np.std(train_y))
    base_loss = np.abs(cal_df["actual"].values - cal_df["base_pred"].values)
    corr_loss = np.abs(cal_df["actual"].values - (cal_df["base_pred"].values + oof))
    valid = ~np.isnan(oof)
    gate_label = ((corr_loss + delta) < base_loss).astype(int)

    H = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    H.fit(X_cal, y_cal)

    G = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(C=LOGIT_C, class_weight="balanced", max_iter=1000)),
    ])
    G.fit(X_cal[valid], gate_label[valid])

    X_test = test_df[feature_cols].values
    actual = test_df["actual"].values
    base_pred = test_df["base_pred"].values
    seasonal_pred = test_df["seasonal_naive_pred"].values
    local_pred = test_df["local_ets_pred"].values

    pred_err = H.predict(X_test)
    gate_prob = G.predict_proba(X_test)[:, 1]
    clip_scale = np.maximum(test_df["r_std_12"].values, 1e-6)
    pred_err_clip = np.clip(
        pred_err,
        -CLIP_KAPPA * clip_scale,
        CLIP_KAPPA * clip_scale,
    )
    gate_weight = np.where(gate_prob > GATE_THRESHOLD, gate_prob, 0.0)
    corr_pred = base_pred + gate_weight * pred_err_clip

    results = [
        metrics_row("Frozen base SARIMA", actual, base_pred, mase_denom),
        metrics_row("Frozen correction + gate", actual, corr_pred, mase_denom),
    ]
    return pd.DataFrame(results)

# ============================================================
# METHOD 3: KALMAN BIAS-CORRECTED
# ============================================================
def kalman_run(base_pred, actual, q, r, init_c=0.0, init_p=1.0, return_state=False):
    n = len(actual)
    preds = np.zeros(n)
    c = init_c
    P = init_p
    cs = []
    Ps = []

    for t in range(n):
        c_pred = c
        P_pred = P + q
        preds[t] = base_pred[t] + c_pred

        resid = actual[t] - base_pred[t]
        K = P_pred / (P_pred + r)
        c = c_pred + K * (resid - c_pred)
        P = (1 - K) * P_pred

        cs.append(c)
        Ps.append(P)

    if return_state:
        return preds, (c, P, np.array(cs), np.array(Ps))
    return preds

def run_kalman_bias_corrected(ctx):
    dates = ctx["dates"]
    y = ctx["y"]
    train_end_idx = ctx["train_end_idx"]
    cal_end_idx = ctx["cal_end_idx"]
    cal_start_idx = ctx["cal_start_idx"]
    test_start_idx = ctx["test_start_idx"]
    mase_denom = ctx["mase_denom"]

    train_y = y[:train_end_idx + 1]
    _, base_params = fit_filter(train_y, BASE_ORDER, BASE_SEASONAL_ORDER, BASE_TREND, maxiter=30)
    frozen_forecast = make_frozen_forecaster(BASE_ORDER, BASE_SEASONAL_ORDER, BASE_TREND, base_params)

    rows = []
    for i in range(cal_start_idx, len(y)):
        hist = y[:i]
        b = frozen_forecast(hist)
        rows.append({
            "idx": i,
            "date": dates.iloc[i],
            "actual": y[i],
            "base_pred": b,
        })

    roll_df = pd.DataFrame(rows)
    cal_df = roll_df[roll_df["idx"] <= cal_end_idx].reset_index(drop=True)
    test_df = roll_df[roll_df["idx"] >= test_start_idx].reset_index(drop=True)

    best = None
    for q in Q_GRID:
        for r in R_GRID:
            preds, state = kalman_run(
                cal_df["base_pred"].values,
                cal_df["actual"].values,
                q, r,
                return_state=True,
            )
            mae = float(np.mean(np.abs(cal_df["actual"].values - preds)))
            if best is None or mae < best[0]:
                best = (mae, q, r, state[0], state[1])

    _, best_q, best_r, c_end, p_end = best
    preds = kalman_run(
        test_df["base_pred"].values,
        test_df["actual"].values,
        best_q, best_r,
        init_c=c_end, init_p=p_end,
        return_state=False,
    )

    return pd.DataFrame([
        metrics_row("Kalman bias-corrected", test_df["actual"].values, preds, mase_denom)
    ])

# ============================================================
# METHOD 4 + 5: ADAPTIVE BASE / ADAPTIVE CORRECTION + REFRESH
# ============================================================
def run_adaptive_methods(ctx):
    dates = ctx["dates"]
    y = ctx["y"]
    train_end_idx = ctx["train_end_idx"]
    cal_end_idx = ctx["cal_end_idx"]
    cal_start_idx = ctx["cal_start_idx"]
    test_start_idx = ctx["test_start_idx"]
    mase_denom = ctx["mase_denom"]

    train_y = y[:train_end_idx + 1]
    base_fit, base_params = fit_filter(train_y, BASE_ORDER, BASE_SEASONAL_ORDER, BASE_TREND, maxiter=30)
    frozen_forecast = make_frozen_forecaster(BASE_ORDER, BASE_SEASONAL_ORDER, BASE_TREND, base_params)

    train_resid = np.asarray(base_fit.resid, dtype=float)
    resid_map = {i: float(train_resid[i]) for i in range(len(train_resid))}

    # Train article-style correction/gate on calibration
    rows = []
    for i in range(cal_start_idx, len(y)):
        history = y[:i]
        b = frozen_forecast(history)
        s = seasonal_naive(history, SEASONAL_PERIOD)
        l = local_ets_forecast(history, LOCAL_WINDOW, SEASONAL_PERIOD)
        feat = build_feature_row(dates, history, i, b, s, l, resid_map)
        err = float(y[i] - b)
        row = dict(feat)
        row.update({
            "idx": i,
            "date": dates.iloc[i],
            "actual": y[i],
            "base_pred": b,
            "base_error": err,
        })
        rows.append(row)
        resid_map[i] = err

    roll_df = pd.DataFrame(rows)
    cal_df = roll_df[roll_df["idx"] <= cal_end_idx].reset_index(drop=True)

    feature_cols = [
        c for c in cal_df.columns
        if c not in ["idx", "date", "actual", "base_pred", "base_error"]
    ]

    X_cal = cal_df[feature_cols].values
    y_cal = cal_df["base_error"].values

    oof = np.full(len(cal_df), np.nan)
    for i in range(MIN_OOF_TRAIN, len(cal_df)):
        mdl = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ])
        mdl.fit(X_cal[:i], y_cal[:i])
        oof[i] = mdl.predict(X_cal[i:i+1])[0]

    delta = DELTA_SCALE * float(np.std(train_y))
    base_loss = np.abs(cal_df["actual"].values - cal_df["base_pred"].values)
    corr_loss = np.abs(cal_df["actual"].values - (cal_df["base_pred"].values + oof))
    valid = ~np.isnan(oof)
    gate_label = ((corr_loss + delta) < base_loss).astype(int)

    H = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    H.fit(X_cal, y_cal)

    G = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(C=LOGIT_C, class_weight="balanced", max_iter=1000)),
    ])
    G.fit(X_cal[valid], gate_label[valid])

    cal_gate_prob = G.predict_proba(X_cal)[:, 1]
    cal_corr = (
        cal_df["base_pred"].values
        + np.where(cal_gate_prob > GATE_THRESHOLD, cal_gate_prob, 0.0)
        * np.clip(H.predict(X_cal), -np.maximum(cal_df["r_std_12"].values, 1e-6), np.maximum(cal_df["r_std_12"].values, 1e-6))
    )
    cal_proposed_mae = float(np.mean(np.abs(cal_df["actual"].values - cal_corr)))

    # Adaptive deployment
    res_current, _ = fit_filter(y[:cal_end_idx + 1], BASE_ORDER, BASE_SEASONAL_ORDER, BASE_TREND, maxiter=20)
    last_refit_idx = cal_end_idx

    resid_hist = {i: float(train_resid[i]) for i in range(len(train_resid))}
    for _, row in cal_df.iterrows():
        resid_hist[int(row["idx"])] = float(row["base_error"])

    recent_corr = deque(np.abs(cal_df["actual"].values - cal_corr)[-ROLL_WINDOW:].tolist(), maxlen=ROLL_WINDOW)

    actuals = []
    base_preds = []
    corr_preds = []

    for i in range(test_start_idx, len(y)):
        history = y[:i]
        b = float(np.asarray(res_current.forecast(1))[0])
        s = seasonal_naive(history, SEASONAL_PERIOD)
        l = local_ets_forecast(history, LOCAL_WINDOW, SEASONAL_PERIOD)

        feat = build_feature_row(dates, history, i, b, s, l, resid_hist)
        x = pd.DataFrame([feat])[feature_cols].values

        pe = float(H.predict(x)[0])
        gp = float(G.predict_proba(x)[0, 1])
        clip_scale = max(feat["r_std_12"], 1e-6)
        pe = float(np.clip(pe, -CLIP_KAPPA * clip_scale, CLIP_KAPPA * clip_scale))
        weight = gp if gp > GATE_THRESHOLD else 0.0
        yhat = b + weight * pe

        actual = y[i]
        ce = abs(actual - yhat)

        actuals.append(actual)
        base_preds.append(b)
        corr_preds.append(yhat)

        resid_hist[i] = actual - b
        recent_corr.append(ce)
        res_current = res_current.extend([actual])

        trigger = (
            (len(recent_corr) == ROLL_WINDOW and np.mean(recent_corr) > TRIGGER_MULT * cal_proposed_mae)
            or ((i - last_refit_idx) >= MAX_AGE)
        )
        if trigger:
            res_current, _ = fit_filter(y[:i + 1], BASE_ORDER, BASE_SEASONAL_ORDER, BASE_TREND, maxiter=15)
            last_refit_idx = i

    actuals = np.array(actuals)
    base_preds = np.array(base_preds)
    corr_preds = np.array(corr_preds)

    return pd.DataFrame([
        metrics_row("Adaptive base SARIMA", actuals, base_preds, mase_denom),
        metrics_row("Adaptive correction + refresh", actuals, corr_preds, mase_denom),
    ])

# ============================================================
# MAIN
# ============================================================
def main():
    ctx = load_data(DATA_PATH)

    results = pd.concat([
        run_adaptive_methods(ctx),
        run_frozen_methods(ctx),
        run_kalman_bias_corrected(ctx),
    ], ignore_index=True)

    results = results.sort_values("MAE").reset_index(drop=True)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "five_methods_results.csv"
    results.to_csv(outpath, index=False)

    print(results)
    print(f"\nSaved results to: {outpath.resolve()}")

if __name__ == "__main__":
    main()
