# tools/sensor_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# WHY NO REAL DATASET?
# ─────────────────────────────────────────────────────────────────────────────
# This project uses SIMULATED sensor data instead of a CSV/database dataset.
# Reason: Real greenhouse IoT sensors stream live readings every few seconds.
# In an academic project without hardware, we simulate that stream using:
#   1. random.gauss() — mimics natural sensor fluctuation (Gaussian noise)
#   2. time-of-day drift — morning light increases, noon temp peaks, etc.
#   3. scenario injection — occasionally inject "crisis" readings to test agents
#
# This is standard practice in IoT research papers and academic projects.
# ─────────────────────────────────────────────────────────────────────────────

import random
import time
from datetime import datetime
from config import THRESHOLDS


# ── Simulated historical baseline (acts like a "dataset") ────────────────────
# These represent typical greenhouse baselines for a tomato crop.
BASELINE = {
    "temperature":   {"mean": 26.0, "std": 3.5},
    "humidity":      {"mean": 60.0, "std": 8.0},
    "soil_moisture": {"mean": 52.0, "std": 10.0},
    "light_level":   {"mean": 600.0,"std": 120.0},
    "co2_level":     {"mean": 700.0, "std": 80.0},
}

# ── Crisis scenarios (inject randomly to test alert + control agents) ─────────
SCENARIOS = [
    {"name": "heat_wave",       "temperature": 40.0, "humidity": 28.0},
    {"name": "drought",         "soil_moisture": 12.0},
    {"name": "overwatering",    "soil_moisture": 92.0},
    {"name": "night_cycle",     "light_level": 50.0,  "temperature": 16.0},
    {"name": "co2_spike",       "co2_level": 1500.0},
    {"name": "normal",          },   # no override — fully normal cycle
]


def _time_drift() -> dict:
    """Simulate how readings shift based on time of day."""
    hour = datetime.now().hour
    # Morning (6–10): light rising, temp moderate
    # Noon   (10–15): max light, max temp
    # Evening(15–20): declining
    # Night  (20–6):  dark, cool
    if 6 <= hour < 10:
        return {"light_level": 0.6, "temperature": 0.85}
    elif 10 <= hour < 15:
        return {"light_level": 1.0, "temperature": 1.0}
    elif 15 <= hour < 20:
        return {"light_level": 0.7, "temperature": 0.9}
    else:
        return {"light_level": 0.05, "temperature": 0.7}


def _read_sensor(name: str, override: float = None) -> float:
    """Generate a realistic sensor value using Gaussian noise around the baseline."""
    if override is not None:
        # add slight noise even to injected crisis values
        return round(override + random.gauss(0, 0.5), 2)

    drift = _time_drift()
    base = BASELINE[name]
    value = random.gauss(base["mean"], base["std"])
    # Apply time-of-day multiplier where relevant
    if name in drift:
        value *= drift[name]
    return round(value, 2)


def get_sensor_readings(force_scenario: str = None) -> dict:
    """
    Main function called by the Monitor Agent.
    Returns a full dict of all sensor readings.
    Optionally force a named scenario for testing.
    """
    # Pick a scenario (10% chance of crisis, else normal)
    if force_scenario:
        scenario = next((s for s in SCENARIOS if s["name"] == force_scenario), {})
    else:
        weights = [5, 5, 5, 5, 5, 75]   # 25% crisis, 75% normal
        scenario = random.choices(SCENARIOS, weights=weights, k=1)[0]

    readings = {
        "temperature":   _read_sensor("temperature",   scenario.get("temperature")),
        "humidity":      _read_sensor("humidity",      scenario.get("humidity")),
        "soil_moisture": _read_sensor("soil_moisture", scenario.get("soil_moisture")),
        "light_level":   _read_sensor("light_level",  scenario.get("light_level")),
        "co2_level":     _read_sensor("co2_level",    scenario.get("co2_level")),
        "_scenario":     scenario.get("name", "normal"),
    }
    return readings


def classify_reading(sensor: str, value: float) -> str:
    """Returns 'OK', 'HIGH', or 'LOW' for a given sensor reading."""
    lo = THRESHOLDS[sensor]["min"]
    hi = THRESHOLDS[sensor]["max"]
    if value < lo:
        return "LOW"
    elif value > hi:
        return "HIGH"
    return "OK"
