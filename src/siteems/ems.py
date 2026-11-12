"""Energy management of the site: battery, cooling plant used as thermal storage, PV; model predictive control.

Every hour a linear program (SciPy, HiGHS) plans charge, discharge, cooling power, PV curtailment, import and export
from now to the end of the day, or to the end of the next day once its prices are known (from 13:00), minimising
  energy cost (day-ahead price + fixed per-kWh charges) - export revenue (day-ahead price)
  + demand charge x rise of the billing-period peak + battery wear.
Only the first hour is carried out, with the real load, cooling demand and PV; a local controller then discharges
the battery further if the real import would exceed the planned peak. Forecasts are corrected every hour with the
latest observed error (AR(1) decay).

Cooling plant: the chiller's electric power f can differ from the electric-equivalent cooling demand c as long as a
cold store (thermal mass, buffer) stays within 0..S; S = 0 means no flexibility (f = c). The same model describes a
heat pump with a buffer tank.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class Site:
    batt_kwh: float = 200.0
    batt_kw: float = 100.0
    eta: float = 0.95                 # one way; 90% round trip
    soc_min: float = 0.1
    soc_start: float = 0.5
    cool_kw: float = 40.0             # chiller rating (max metered cooling power is 37 kW)
    cool_store_kwh: float = 40.0      # electric-equivalent cold store
    adder: float = 0.15               # EUR/kWh on top of the day-ahead price for imports (grid fees, levies, taxes)
    demand_charge: float = 100.0      # EUR per kW of the billing-period peak and year
    billing: str = "annual"           # or "monthly"
    wear: float = 0.03                # EUR per kWh discharged
    peak_floor: float = 0.0           # the plan never targets a peak below this (keeps energy for the large peaks)

    @property
    def dc_rate(self) -> float:
        return self.demand_charge if self.billing == "annual" else self.demand_charge / 12


def solve(base, pv, cool, price, soc0, s0, peak0, site: Site, base_peak=None):
    """Plan over len(base) hours. base: non-cooling load, cool: cooling demand (both kW, forecasts), price EUR/MWh.
    base_peak: optional higher load forecast used only in the peak constraint. Returns dict of first-hour decisions."""
    H = len(base)
    n = 8 * H + 3                  # ... + peak + two slacks for the end-of-horizon targets
    ch, dis, f, imp, exp, cur, soc, s = (np.arange(H) + k * H for k in range(8))
    P = 8 * H
    c = np.zeros(n)
    p = np.asarray(price) / 1000
    c[imp] = p + site.adder
    c[exp] = -p
    c[dis] = site.wear
    c[cur] = 1e-4
    c[P] = site.dc_rate
    sb, sc = P + 1, P + 2          # shortfall below the end targets (battery, cold store), priced at 0.5 EUR/kWh
    c[sb] = c[sc] = 0.5
    rows, cols, vals, beq = [], [], [], []
    r = 0
    for t in range(H):             # balance: imp - exp - f - ch + dis - cur = base - pv
        for j, v in ((imp[t], 1), (exp[t], -1), (f[t], -1), (ch[t], -1), (dis[t], 1), (cur[t], -1)):
            rows.append(r); cols.append(j); vals.append(v)
        beq.append(base[t] - pv[t]); r += 1
    for t in range(H):             # battery
        rows += [r, r, r]; cols += [soc[t], ch[t], dis[t]]; vals += [1, -site.eta, 1 / site.eta]
        if t > 0:
            rows.append(r); cols.append(soc[t - 1]); vals.append(-1)
        beq.append(soc0 if t == 0 else 0.0); r += 1
    for t in range(H):             # cold store
        rows += [r, r]; cols += [s[t], f[t]]; vals += [1, -1]
        if t > 0:
            rows.append(r); cols.append(s[t - 1]); vals.append(-1)
        beq.append((s0 if t == 0 else 0.0) - cool[t]); r += 1
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    rows, cols, vals, bub = [], [], [], []
    r = 0
    for t in range(H):             # import within the peak
        rows += [r, r]; cols += [imp[t], P]; vals += [1, -1]; bub.append(0.0); r += 1
    if base_peak is not None:      # robust peak: higher load forecast + f + ch - dis - (pv - cur) <= P
        for t in range(H):
            for j, v in ((f[t], 1), (ch[t], 1), (dis[t], -1), (cur[t], 1), (P, -1)):
                rows.append(r); cols.append(j); vals.append(v)
            bub.append(pv[t] - base_peak[t]); r += 1
    rows += [r, r]; cols += [soc[-1], sb]; vals += [-1, -1]           # soc_end + shortfall >= target
    bub.append(-min(site.soc_start, 1.0) * site.batt_kwh); r += 1
    rows += [r, r]; cols += [s[-1], sc]; vals += [-1, -1]
    bub.append(-0.5 * site.cool_store_kwh); r += 1
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    lo, hi = np.zeros(n), np.full(n, np.inf)
    hi[ch] = hi[dis] = site.batt_kw
    hi[f] = site.cool_kw
    hi[cur] = np.maximum(pv, 0)
    lo[soc], hi[soc] = site.soc_min * site.batt_kwh, site.batt_kwh
    hi[s] = site.cool_store_kwh
    lo[P] = max(peak0, site.peak_floor)
    res = linprog(c, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=beq, bounds=np.column_stack([lo, hi]), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    x = res.x
    return {"ch": x[ch[0]], "dis": x[dis[0]], "f": x[f[0]], "cur": x[cur[0]], "peak": x[P]}


def realise(plan, load, cool, pv, price, soc, s, peak, site: Site):
    """Carry out one hour with the realised load, cooling demand and PV; returns the new state and the hour's flows."""
    f = float(np.clip(plan["f"], max(0.0, cool - s), min(site.cool_kw, site.cool_store_kwh - s + cool)))
    ch, dis = plan["ch"], plan["dis"]
    ch = min(ch, (site.batt_kwh - soc) / site.eta)
    dis = min(dis, (soc - site.soc_min * site.batt_kwh) * site.eta)
    net = load - cool + f - pv + ch - dis
    cap = max(plan["peak"], peak)
    if net > cap:                  # local peak guard: stop charging, then discharge more
        cut = min(ch, net - cap)
        ch -= cut; net -= cut
        extra = min(net - cap, site.batt_kw - dis, (soc - site.soc_min * site.batt_kwh) * site.eta - dis)
        if extra > 0:
            dis += extra; net -= extra
    soc = soc + site.eta * ch - dis / site.eta
    s = s + f - cool
    imp, exp = max(net, 0.0), max(-net, 0.0)
    if price < 0:
        exp = 0.0                  # curtail PV instead of exporting at a negative price
    return soc, s, {"imp": imp, "exp": exp, "ch": ch, "dis": dis, "f": f}


def horizon_end(i: int, hours_local: np.ndarray, n: int) -> int:
    """Index after the last hour of the planning horizon starting at i (local hour hours_local[i])."""
    h = hours_local[i]
    end = i + (24 - h) + (24 if h >= 13 else 0)
    return min(end, n)


def correct(fc: np.ndarray, err: float, phi: float) -> np.ndarray:
    return fc + err * phi ** np.arange(1, len(fc) + 1)


def simulate(d: pd.DataFrame, fc: pd.DataFrame | None, site: Site, strategy: str = "mpc", phi: float = 0.8,
             robust: bool = False, threshold: float = 0.0) -> pd.DataFrame:
    """d: realised hourly data of the test period (load, cooling, pv, price, hour, month); fc: forecasts with the same
    index (load, cooling, pv; load_q90 when robust). strategy: mpc, rule (self-consumption), threshold (peak-shaving
    rule at a fixed import threshold) or none (no battery, no cooling flexibility)."""
    n = len(d)
    load, cool, pv, price = (d[c].to_numpy(float) for c in ("load", "cooling", "pv", "price"))
    period = d["month"].to_numpy() if site.billing == "monthly" else np.zeros(n, int)
    hours = d["hour"].to_numpy()
    soc, s, peak = site.soc_start * site.batt_kwh, 0.5 * site.cool_store_kwh, 0.0
    out = []
    err = {"base": 0.0, "cool": 0.0, "pv": 0.0}
    for i in range(n):
        if i > 0 and period[i] != period[i - 1]:
            peak = 0.0
        if strategy == "none":
            net = load[i] - pv[i]
            fl = {"imp": max(net, 0), "exp": 0.0 if price[i] < 0 else max(-net, 0), "ch": 0.0, "dis": 0.0, "f": cool[i]}
        elif strategy == "rule":
            net = load[i] - pv[i]
            ch = min(max(-net, 0), site.batt_kw, (site.batt_kwh - soc) / site.eta)
            dis = min(max(net, 0), site.batt_kw, (soc - site.soc_min * site.batt_kwh) * site.eta)
            soc += site.eta * ch - dis / site.eta
            net += ch - dis
            fl = {"imp": max(net, 0), "exp": 0.0 if price[i] < 0 else max(-net, 0), "ch": ch, "dis": dis, "f": cool[i]}
        elif strategy == "threshold":
            # peak-shaving rule: discharge above the threshold, recharge below threshold - margin, keep the battery full
            net = load[i] - pv[i]
            avail = (soc - site.soc_min * site.batt_kwh) * site.eta
            dis = min(max(net - threshold, 0), site.batt_kw, avail)
            ch = 0.0 if dis > 0 else min(max(threshold - 20 - net, 0), site.batt_kw, (site.batt_kwh - soc) / site.eta)
            soc += site.eta * ch - dis / site.eta
            net += ch - dis
            fl = {"imp": max(net, 0), "exp": 0.0 if price[i] < 0 else max(-net, 0), "ch": ch, "dis": dis, "f": cool[i]}
        else:
            j = horizon_end(i, hours, n)
            cool_fc = np.clip(correct(fc["cooling"].to_numpy()[i:j], err["cool"], phi), 0, site.cool_kw)
            base_fc = correct(fc["load"].to_numpy()[i:j] - fc["cooling"].to_numpy()[i:j], err["base"], phi)
            pv_fc = np.clip(correct(fc["pv"].to_numpy()[i:j], err["pv"], phi), 0, None)
            bp = None
            if robust:
                bp = np.maximum(base_fc, fc["load_q90"].to_numpy()[i:j] - fc["cooling"].to_numpy()[i:j] + err["base"])
            plan = solve(base_fc, pv_fc, cool_fc, price[i:j], soc, s, peak, site, bp)
            soc, s, fl = realise(plan, load[i], cool[i], pv[i], price[i], soc, s, peak, site)
            if fc is not None:     # error of the hour just observed, used to correct the next plans
                err = {"base": (load[i] - cool[i]) - (fc["load"].iat[i] - fc["cooling"].iat[i]),
                       "cool": cool[i] - fc["cooling"].iat[i], "pv": pv[i] - fc["pv"].iat[i]}
        peak = max(peak, fl["imp"])
        out.append({**fl, "soc": soc, "cold_store": s})
    return pd.DataFrame(out, index=d.index)


def bill(d: pd.DataFrame, flows: pd.DataFrame, site: Site) -> dict:
    p = d["price"].to_numpy() / 1000
    energy = float((flows["imp"] * (p + site.adder)).sum())
    export = float((flows["exp"] * p).sum())
    if site.billing == "annual":
        peaks = [flows["imp"].max()]
        demand = site.demand_charge * peaks[0] * len(d) / 8760
    else:
        peaks = flows["imp"].groupby(d["month"].to_numpy()).max().tolist()
        demand = site.demand_charge / 12 * sum(peaks)
    wear = site.wear * float(flows["dis"].sum())
    return {"energy_eur": energy, "export_eur": export, "demand_eur": demand, "wear_eur": wear,
            "total_eur": energy - export + demand + wear, "peak_kw": float(max(peaks)), "mean_peak_kw": float(np.mean(peaks)),
            "import_mwh": float(flows["imp"].sum() / 1000), "export_mwh": float(flows["exp"].sum() / 1000),
            "cycles": float(flows["dis"].sum() / site.batt_kwh) if site.batt_kwh else 0.0}


def with_assets(site: Site, battery: bool = True, cooling: bool = True) -> Site:
    s = site
    if not battery:
        s = replace(s, batt_kwh=0.0, batt_kw=0.0)
    if not cooling:
        s = replace(s, cool_store_kwh=0.0)
    return s
