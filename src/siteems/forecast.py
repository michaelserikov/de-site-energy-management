"""Day-ahead forecasts of site load, cooling demand and PV, issued at noon of the day before.

Information at the issue time (12:00 local on D-1): measurements up to that hour, the calendar and, in the weather
variant, ERA5 weather for day D. ERA5 is a reanalysis, so it stands in for a numerical weather forecast and makes
that variant optimistic. Models are refitted at the start of every month on all data before it (expanding window).

Variants
  naive       load and cooling: same hour one week earlier; PV: mean of the same hour over the last 7 known days
  ml          gradient boosting on calendar and lagged measurements (no weather)
  ml_weather  the same plus temperature and irradiance of the target hour
  perfect     the realised values
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 3
TZ = "Europe/Berlin"
CAL = ["hour", "dow", "workday", "holiday", "doy_sin", "doy_cos"]
WX = ["temp", "ghi", "dir", "dif", "cloud", "temp_24h"]
TARGETS = ("load", "cooling", "pv")


def features(d: pd.DataFrame, target: str) -> pd.DataFrame:
    """Lags are at least 48 hours: day D-2 is complete at noon of D-1, so every lag is known at the issue time."""
    y = d[target]
    lag48 = y.shift(48)
    f = pd.DataFrame(index=d.index)
    for c in ("hour", "dow", "workday", "holiday"):
        f[c] = d[c]
    f["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 365.25)
    f["lag48"] = lag48
    f["lag168"] = y.shift(168)
    f["lag336"] = y.shift(336)
    f["mean_7d"] = y.shift(48).rolling(168, min_periods=100).mean()
    # same hour on the latest earlier day of the same type (workday or not), at least two days back
    same = y.copy() * np.nan
    wd = d["workday"].to_numpy()
    for k in range(2, 9):
        cand = y.shift(24 * k).where(pd.Series(wd, index=d.index).shift(24 * k) == wd)
        same = same.fillna(cand)
    f["same_type"] = same
    for c in WX:
        if c in d:
            f[c] = d[c]
    return f


def _model(quantile: float | None = None):
    if quantile is None:
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                             l2_regularization=1.0, random_state=SEED)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=40, random_state=SEED)


def naive(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    out["load"] = d["load"].shift(168)
    out["cooling"] = d["cooling"].shift(168)
    by_hour = [d["pv"].shift(24 * k) for k in range(2, 9)]
    out["pv"] = pd.concat(by_hour, axis=1).mean(axis=1)
    return out


def rolling_ml(d: pd.DataFrame, test_start: str, weather: bool, quantiles: tuple[float, ...] = ()) -> pd.DataFrame:
    """Monthly refits; returns forecasts for every hour from test_start (columns per target, plus load_q90 etc.)."""
    d = d.copy()
    d["temp_24h"] = d["temp"].rolling(24, min_periods=1).mean()
    out = pd.DataFrame(index=d.index[d.index >= test_start])
    first = pd.Timestamp(test_start, tz="UTC").tz_convert(TZ).normalize()        # months in local time
    months = pd.date_range(first, d.index[-1].tz_convert(TZ), freq="MS").tz_convert("UTC")
    for target in TARGETS:
        X = features(d, target)
        if not weather:
            X = X.drop(columns=[c for c in WX if c in X])
        if target == "pv":           # PV needs no load-type calendar, but irradiance; without weather only lags
            X = X.drop(columns=["dow", "workday", "holiday"])
        y = d[target]
        for i, m0 in enumerate(months):
            m1 = months[i + 1] if i + 1 < len(months) else d.index[-1] + pd.Timedelta(hours=1)
            # issue time of the first day of the month is noon of the previous day: train on data before it
            cut = m0 - pd.Timedelta(hours=12)
            tr = (X.index < cut) & X.notna().all(axis=1) & y.notna()
            te = (X.index >= m0) & (X.index < m1)
            mdl = _model().fit(X[tr], y[tr])
            out.loc[X.index[te], target] = mdl.predict(X[te].fillna(X[tr].median()))
            for q in quantiles:
                mq = _model(q).fit(X[tr], y[tr])
                out.loc[X.index[te], f"{target}_q{int(q * 100)}"] = mq.predict(X[te].fillna(X[tr].median()))
    out["pv"] = out["pv"].clip(lower=0)
    out.loc[d.loc[out.index, "ghi"].to_numpy() <= 0, "pv"] = 0.0
    out["cooling"] = out["cooling"].clip(lower=0)
    return out


def scores(actual: pd.DataFrame, fc: pd.DataFrame) -> dict:
    res = {}
    for t in TARGETS:
        e = (fc[t] - actual[t]).dropna()
        res[f"{t}_mae"] = float(e.abs().mean())
        res[f"{t}_rmse"] = float(np.sqrt((e ** 2).mean()))
    a = actual["load"].groupby(actual["date"]).max()
    f = fc["load"].groupby(actual["date"]).max()
    res["daily_peak_mae"] = float((f - a).abs().mean())
    return res
