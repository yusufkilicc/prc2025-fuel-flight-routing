"""
Smoke tests for the route-decision engine (no data/server needed).
Run:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import route_decision as rd  # noqa: E402

# IST, JFK, AMS coords (deg)
IST = (41.262, 28.741)
JFK = (40.640, -73.779)
AMS = (52.309, 4.764)


def test_scenario_routes_shape():
    rows = rd.scenario_routes(*IST, *JFK, "A359")
    assert len(rows) == 3
    labels = {r["label"] for r in rows}
    assert labels == {"Great-circle", "Wind / fuel-optimal", "Published ATC route"}
    assert sum(r["recommended"] for r in rows) == 1  # exactly one recommended


def test_fuel_is_physically_sane():
    # IST->JFK on an A350 burns roughly 40-70 t; allow a generous band.
    rows = rd.scenario_routes(*IST, *JFK, "A359")
    rec = next(r for r in rows if r["recommended"])
    assert 35_000 < rec["fuel_kg"] < 80_000
    assert rec["time_min"] > 300  # > 5 h for a transatlantic
    assert rec["dist_km"] > 7000


def test_uncertainty_band_brackets_estimate():
    rows = rd.scenario_routes(*IST, *JFK, "A359")
    for r in rows:
        assert r["fuel_lo"] <= r["fuel_kg"] <= r["fuel_hi"]


def test_co2_factor():
    rows = rd.scenario_routes(*IST, *JFK, "A20N")
    r = rows[0]
    assert abs(r["co2_kg"] - r["fuel_kg"] * rd.CO2_PER_KG_FUEL) < 5  # rounding only


def test_both_uis_share_load_factor():
    # A2 regression guard: score_path and scenario_routes must use the same default.
    import inspect
    sp = inspect.signature(rd.score_path).parameters["load_factor"].default
    sr = inspect.signature(rd.scenario_routes).parameters["load_factor"].default
    assert sp == sr == rd.LOAD_FACTOR


def test_class_calibration_distinct():
    assert rd._class_calib("A359") == rd.CLASS_CALIB["wide"]
    assert rd._class_calib("A20N") == rd.CLASS_CALIB["narrow"]


def test_deviate_changes_path_but_keeps_endpoints():
    base = rd.great_circle(*IST, *JFK, n=40)
    dev = rd.deviate(base, 300.0)
    assert dev.shape == base.shape
    # endpoints unchanged
    assert abs(dev[0][0] - base[0][0]) < 1e-6
    assert abs(dev[-1][0] - base[-1][0]) < 1e-6
    # middle moved
    mid = len(base) // 2
    assert abs(dev[mid][0] - base[mid][0]) + abs(dev[mid][1] - base[mid][1]) > 0.1
