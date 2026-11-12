"""Forecasts, energy management runs and the value of forecasts and flexibility.

    python -m siteems.run            # all runs, about 20 minutes
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import data, ems, forecast

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "figures"
TEST_START, TEST_END = "2016-05-31 22:00", "2017-05-31 21:00"     # 1 June 2016 to 31 May 2017, local time
PV_KWP = 150.0


def prepare() -> pd.DataFrame:
    d = data.load_all()
    cap = d["pv_meas"].quantile(0.999)                  # measured array scaled to a hypothetical 150 kWp
    d["pv"] = (d["pv_meas"] / cap * PV_KWP).clip(lower=0)
    return d


def run_case(name, d, fc, site, strategy="mpc", robust=False, threshold=0.0):
    t = time.time()
    fl = ems.simulate(d, fc, site, strategy=strategy, robust=robust, threshold=threshold)
    b = ems.bill(d, fl, site)
    print(f"{name:45s} total {b['total_eur']:10.0f}  peak {b['peak_kw']:6.1f}  ({time.time() - t:.0f} s)", flush=True)
    return {"case": name, **b}, fl


def main() -> dict:
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    d = prepare()
    info = {"hours": len(d), "interpolated_hours": int(d["interpolated"].sum()),
            "mean_load_kw": float(d["load"].mean()), "max_load_kw": float(d["load"].max()),
            "pv_scale_kw_per_measured": float(PV_KWP / d["pv_meas"].quantile(0.999))}
    test = d.loc[TEST_START:TEST_END]
    cache = RES / "forecasts.csv.gz"
    if cache.exists():             # the monthly refits take several minutes; delete the file to recompute
        allf = pd.read_csv(cache, header=[0, 1], index_col=0)
        allf.index = pd.to_datetime(allf.index, utc=True)
        fcs = {k: allf[k] for k in allf.columns.get_level_values(0).unique()}
    else:
        t0 = time.time()
        fcs = {"naive": forecast.naive(d),
               "ml": forecast.rolling_ml(d, TEST_START, weather=False, quantiles=(0.9,)),
               "ml_weather": forecast.rolling_ml(d, TEST_START, weather=True, quantiles=(0.9,))}
        print(f"forecasts {time.time() - t0:.0f} s", flush=True)
        fcs = {k: v.loc[test.index] for k, v in fcs.items()}
        pd.concat(fcs, axis=1).to_csv(cache, float_format="%.3f")
    fcs = {k: v.loc[test.index] for k, v in fcs.items()}
    for k, v in fcs.items():
        info[f"missing_{k}"] = int(v.isna().sum().sum())
        fcs[k] = v.fillna(test[["load", "cooling", "pv"]].shift(168).reindex(columns=v.columns)).ffill()
    fcs["perfect"] = test[["load", "cooling", "pv"]]
    sc = pd.DataFrame({k: forecast.scores(test, v) for k, v in fcs.items() if k != "perfect"}).T
    sc.index.name = "forecast"
    sc.to_csv(RES / "forecast_scores.csv", float_format="%.3f")
    print(sc.round(2).to_string(), flush=True)

    site = ems.Site()
    tr = d.loc[:TEST_START].iloc[:-1]
    thr = float((tr["load"] - tr["pv"]).quantile(0.999))
    info["threshold_kw"] = thr
    rows, flows = [], {}
    cases = [("no battery, no flexibility", None, site, "none", False),
             ("rule: PV self-consumption", None, site, "rule", False),
             ("rule: peak threshold (training 99.9th percentile)", None, site, "threshold", False),
             ("MPC, naive forecast", fcs["naive"], site, "mpc", False),
             ("MPC, ML forecast", fcs["ml"], site, "mpc", False),
             ("MPC, ML forecast with weather", fcs["ml_weather"], site, "mpc", False),
             ("MPC, ML with weather, P90 peak", fcs["ml_weather"], site, "mpc", True),
             ("MPC, perfect forecast", fcs["perfect"], site, "mpc", False),
             ("MPC, naive forecast, peak floor at the threshold", fcs["naive"], replace(site, peak_floor=thr), "mpc", False),
             ("MPC, ML with weather, peak floor at the threshold", fcs["ml_weather"], replace(site, peak_floor=thr), "mpc", False),
             ("MPC, battery only (ML weather)", fcs["ml_weather"], ems.with_assets(site, cooling=False), "mpc", False),
             ("MPC, cooling flexibility only (ML weather)", fcs["ml_weather"], ems.with_assets(site, battery=False), "mpc", False)]
    for name, fc, s, strat, rob in cases:
        r, fl = run_case(name, test, fc, s, strat, rob, thr)
        rows.append({"billing": "annual", **r})
        flows[name] = fl
    monthly = replace(site, billing="monthly")
    for name, fc, strat, rob in [("no battery, no flexibility", None, "none", False),
                                  ("rule: peak threshold (training 99.9th percentile)", None, "threshold", False),
                                  ("MPC, naive forecast", fcs["naive"], "mpc", False),
                                  ("MPC, ML forecast", fcs["ml"], "mpc", False),
                                  ("MPC, ML forecast with weather", fcs["ml_weather"], "mpc", False),
                                  ("MPC, ML with weather, P90 peak", fcs["ml_weather"], "mpc", True),
                                  ("MPC, perfect forecast", fcs["perfect"], "mpc", False)]:
        r, _ = run_case(name + " [monthly peaks]", test, fc, monthly, strat, rob, thr)
        rows.append({"billing": "monthly", **r, "case": name})
    res = pd.DataFrame(rows)
    for b in ("annual", "monthly"):
        ref = res.loc[(res["billing"] == b) & (res["case"] == "no battery, no flexibility"), "total_eur"].iloc[0]
        res.loc[res["billing"] == b, "saving_eur"] = ref - res.loc[res["billing"] == b, "total_eur"]
    res.to_csv(RES / "ems_results.csv", index=False, float_format="%.2f")
    print(res.round(1).to_string(), flush=True)
    for k, fl in flows.items():
        fl.columns = pd.MultiIndex.from_product([[k], fl.columns])
    pd.concat(flows.values(), axis=1).to_csv(RES / "flows_annual.csv.gz", float_format="%.3f")
    figures(test, fcs, flows, res)
    info["forecast_scores"] = sc.to_dict(orient="index")
    (RES / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def figures(test, fcs, flows, res):
    # one week: load and forecasts
    wk = slice("2016-11-14 00:00", "2016-11-20 23:00")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(test.loc[wk, "load"], color="black", lw=1.5, label="metered load")
    for k, col in (("naive", "#999999"), ("ml", "#1f77b4"), ("ml_weather", "#d62728")):
        ax[0].plot(fcs[k].loc[wk, "load"], lw=1, color=col, label=f"day-ahead, {k.replace('_', ' ')}")
    ax[0].set_ylabel("kW"); ax[0].legend(fontsize=8, ncol=2); ax[0].set_title("Site load and day-ahead forecasts")
    base = flows["no battery, no flexibility"].droplevel(0, axis=1)
    mpc = flows["MPC, ML forecast with weather"].droplevel(0, axis=1)
    ax[1].plot(base.loc[wk, "imp"], color="#999999", lw=1, label="grid import without battery")
    ax[1].plot(mpc.loc[wk, "imp"], color="#d62728", lw=1.2, label="grid import with MPC")
    ax2 = ax[1].twinx()
    ax2.fill_between(mpc.loc[wk].index, mpc.loc[wk, "soc"], color="#2ca02c", alpha=0.15, label="battery energy")
    ax2.set_ylabel("kWh")
    ax[1].set_ylabel("kW"); ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "week_forecast_and_dispatch.png", dpi=130); plt.close(fig)
    # savings by case, annual billing
    r = res[(res["billing"] == "annual") & (res["case"] != "no battery, no flexibility")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ref = res[(res["billing"] == "annual") & (res["case"] == "no battery, no flexibility")].iloc[0]
    parts = {"energy and export": (ref["energy_eur"] - ref["export_eur"]) - (r["energy_eur"] - r["export_eur"]),
             "demand charge": ref["demand_eur"] - r["demand_eur"], "battery wear": -r["wear_eur"]}
    left = np.zeros(len(r))
    for (k, v), col in zip(parts.items(), ("#1f77b4", "#ff7f0e", "#999999")):
        v = v.to_numpy() / 1000
        ax.barh(r["case"], v, left=np.where(v < 0, 0, left), color=col, label=k)
        left = left + np.where(v > 0, v, 0)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("saving against no battery and no flexibility, thousand EUR per year")
    ax.invert_yaxis(); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "savings_by_case.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
