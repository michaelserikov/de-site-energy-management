"""Hourly data of one small industrial site in Konstanz (Open Power System Data, household_data, site industrial3),
day-ahead prices (bidding zone DE-AT-LU, SMARD via energy-charts.info) and ERA5 weather at Konstanz (Open-Meteo).

The OPSD series are cumulative meter readings in kWh; hourly differences are mean power in kW.
Site demand = grid import + on-site PV (the measured PV is far below the load, so all of it is used on site).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SITE = "DE_KN_industrial3_"
START, END = "2016-02-12", "2017-06-03"           # grid import metered from 11 Feb 2016, all series end 4 Jun 2017
TZ = "Europe/Berlin"

# public holidays in Baden-Wuerttemberg
HOLIDAYS = pd.to_datetime([
    "2016-01-01", "2016-01-06", "2016-03-25", "2016-03-28", "2016-05-01", "2016-05-05", "2016-05-16", "2016-05-26",
    "2016-10-03", "2016-11-01", "2016-12-25", "2016-12-26",
    "2017-01-01", "2017-01-06", "2017-04-14", "2017-04-17", "2017-05-01", "2017-05-25", "2017-06-05", "2017-06-15",
    "2017-10-03", "2017-10-31", "2017-11-01", "2017-12-25", "2017-12-26"]).date


def site(raw: Path = RAW) -> pd.DataFrame:
    d = pd.read_csv(raw / "household_data_60min_singleindex.csv", parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    cols = [c for c in d.columns if c.startswith(SITE)]
    kw = d[cols].diff()
    kw.columns = [c.removeprefix(SITE) for c in cols]
    interp = d["interpolated"].fillna("").str.contains(SITE + "grid_import")
    out = pd.DataFrame({
        "pv_meas": kw["pv_roof"] + kw["pv_facade"],
        "load": kw["grid_import"] + kw["pv_roof"] + kw["pv_facade"],
        "cooling": kw["cooling_aggregate"],
        "interpolated": interp,
    })
    out.index = out.index.tz_convert("UTC")
    return out.loc[START:END]


def prices(raw: Path = RAW) -> pd.Series:
    j = json.loads((raw / "prices_de_at_lu_2015_2017.json").read_text(encoding="utf-8"))
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True)
    return pd.Series(j["price"], index=idx, name="price", dtype=float)


def weather(raw: Path = RAW) -> pd.DataFrame:
    j = json.loads((raw / "weather_konstanz_2015_2017.json").read_text(encoding="utf-8"))
    h = j["hourly"]
    idx = pd.to_datetime(h.pop("time")).tz_localize("UTC")
    return pd.DataFrame(h, index=idx).rename(columns={"temperature_2m": "temp", "shortwave_radiation": "ghi",
                                                      "direct_radiation": "dir", "diffuse_radiation": "dif",
                                                      "cloud_cover": "cloud"})


def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    loc = idx.tz_convert(TZ)
    date = loc.date
    hol = np.isin(date, HOLIDAYS)
    xmas = ((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day <= 1))
    dow = loc.day_of_week
    return pd.DataFrame({"hour": loc.hour, "dow": dow, "workday": ((dow < 5) & ~hol & ~xmas).astype(int),
                         "holiday": (hol | xmas).astype(int), "doy": loc.day_of_year, "month": loc.month,
                         "date": pd.to_datetime(date)}, index=idx)


def load_all(raw: Path = RAW) -> pd.DataFrame:
    s = site(raw)
    d = s.join(prices(raw), how="left").join(weather(raw), how="left").join(calendar(s.index))
    d["price"] = d["price"].interpolate(limit=3)
    return d
