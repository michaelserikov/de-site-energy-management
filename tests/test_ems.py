"""Energy management and forecast features on small cases with known answers."""
import numpy as np
import pandas as pd
import pytest

from siteems import ems, forecast


def flat_site(**kw):
    return ems.Site(**{"wear": 0.0, "adder": 0.0, **kw})


def test_battery_shaves_a_single_spike_to_the_lowest_level():
    # base 50 kW, one hour at 150 kW; battery 100 kWh / 100 kW, no losses, starts and must end at 50 kWh.
    # Charging at (P - 50) in the ten hours before the spike must cover the spike above P:
    # 50 + 10 (P - 50) >= 150 - P, so P = 600 / 11; the 13 hours after the spike recharge the battery.
    base = np.full(24, 50.0)
    base[10] = 150.0
    site = flat_site(batt_kwh=100.0, batt_kw=100.0, eta=1.0, soc_min=0.0, cool_store_kwh=0.0, demand_charge=1000.0)
    H = 24
    plan = ems.solve(base, np.zeros(H), np.zeros(H), np.full(H, 30.0), 50.0, 0.0, 0.0, site)
    assert plan["peak"] == pytest.approx(600 / 11, rel=1e-6)


def test_no_cold_store_means_cooling_follows_demand():
    H = 5
    cool = np.array([3.0, 10.0, 0.0, 7.0, 1.0])
    site = flat_site(batt_kwh=0.0, batt_kw=0.0, cool_store_kwh=0.0)
    plan = ems.solve(np.full(H, 20.0), np.zeros(H), cool, np.full(H, 40.0), 0.0, 0.0, 0.0, site)
    assert plan["f"] == pytest.approx(3.0)
    assert plan["peak"] == pytest.approx(30.0)


def test_cold_store_moves_cooling_to_cheap_hours():
    price = np.array([10.0, 200.0, 200.0])
    cool = np.array([0.0, 10.0, 10.0])
    site = flat_site(batt_kwh=0.0, batt_kw=0.0, cool_kw=40.0, cool_store_kwh=20.0, demand_charge=0.0)
    # start empty, end at least half full (10): 30 kWh of cooling in total; the cheap hour takes all the store holds
    plan = ems.solve(np.zeros(3), np.zeros(3), cool, price, 0.0, 0.0, 0.0, site)
    assert plan["f"] == pytest.approx(20.0)


def test_realise_peak_guard_discharges_above_the_planned_peak():
    site = flat_site(batt_kwh=100.0, batt_kw=50.0, eta=1.0, soc_min=0.0, cool_store_kwh=0.0)
    plan = {"ch": 0.0, "dis": 0.0, "f": 0.0, "cur": 0.0, "peak": 80.0}
    soc, s, fl = ems.realise(plan, load=100.0, cool=0.0, pv=0.0, price=50.0, soc=60.0, s=0.0, peak=70.0, site=site)
    assert fl["imp"] == pytest.approx(80.0) and fl["dis"] == pytest.approx(20.0) and soc == pytest.approx(40.0)


def test_realise_keeps_the_cold_store_within_bounds():
    site = flat_site(cool_kw=40.0, cool_store_kwh=10.0)
    plan = {"ch": 0.0, "dis": 0.0, "f": 0.0, "cur": 0.0, "peak": 1e9}
    _, s, fl = ems.realise(plan, load=50.0, cool=15.0, pv=0.0, price=50.0, soc=100.0, s=4.0, peak=0.0, site=site)
    assert fl["f"] == pytest.approx(11.0) and s == pytest.approx(0.0)


def test_horizon_extends_to_next_day_after_noon():
    hours = np.tile(np.arange(24), 3)
    assert ems.horizon_end(0, hours, 72) == 24
    assert ems.horizon_end(12, hours, 72) == 24
    assert ems.horizon_end(13, hours, 72) == 48
    assert ems.horizon_end(60, hours, 72) == 72


def test_bill_prorates_the_annual_demand_charge():
    idx = pd.date_range("2016-06-01", periods=48, freq="h", tz="UTC")
    d = pd.DataFrame({"price": 50.0, "month": 6}, index=idx)
    fl = pd.DataFrame({"imp": 10.0, "exp": 0.0, "dis": 0.0}, index=idx)
    fl.iloc[5, 0] = 30.0
    b = ems.bill(d, fl, ems.Site(adder=0.1, demand_charge=100.0, wear=0.0))
    assert b["demand_eur"] == pytest.approx(100 * 30 * 48 / 8760)
    assert b["energy_eur"] == pytest.approx((47 * 10 + 30) * (0.05 + 0.1))


def test_forecast_lags_use_only_data_two_days_old():
    idx = pd.date_range("2016-01-01", periods=24 * 20, freq="h", tz="UTC")
    d = pd.DataFrame({"load": np.arange(len(idx), dtype=float), "hour": idx.hour, "dow": idx.day_of_week,
                      "workday": (idx.day_of_week < 5).astype(int), "holiday": 0, "doy": idx.day_of_year}, index=idx)
    f = forecast.features(d, "load")
    i = 24 * 15 + 7
    assert f["lag48"].iloc[i] == i - 48
    assert f["lag168"].iloc[i] == i - 168
    assert f["same_type"].iloc[i] <= i - 48


def test_threshold_rule_shaves_above_the_threshold_and_recharges_below():
    idx = pd.date_range("2016-06-01", periods=4, freq="h", tz="UTC")
    d = pd.DataFrame({"load": [100.0, 180.0, 150.0, 60.0], "cooling": 0.0, "pv": 0.0, "price": 40.0,
                      "hour": idx.hour, "month": 6}, index=idx)
    site = flat_site(batt_kwh=100.0, batt_kw=100.0, eta=1.0, soc_min=0.0, soc_start=0.5)
    fl = ems.simulate(d, None, site, strategy="threshold", threshold=160.0)
    assert fl["imp"].tolist() == pytest.approx([140.0, 160.0, 150.0, 90.0])
    assert fl["soc"].tolist() == pytest.approx([90.0, 70.0, 70.0, 100.0])


def test_peak_floor_stops_the_plan_from_shaving_below_it():
    base = np.full(6, 50.0)
    base[3] = 90.0
    site = flat_site(batt_kwh=100.0, batt_kw=100.0, eta=1.0, soc_min=0.0, cool_store_kwh=0.0, peak_floor=80.0)
    plan = ems.solve(base, np.zeros(6), np.zeros(6), np.full(6, 30.0), 50.0, 0.0, 0.0, site)
    assert plan["peak"] == pytest.approx(80.0)
