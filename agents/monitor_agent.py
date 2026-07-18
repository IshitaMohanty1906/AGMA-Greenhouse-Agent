# agents/monitor_agent.py — Reads all sensors, classifies readings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from state import GreenhouseState
from tools.sensor_tools import get_sensor_readings, classify_reading
from config import THRESHOLDS


def monitor_agent(state: GreenhouseState) -> GreenhouseState:
    """
    MONITOR AGENT
    ─────────────
    Role: Read all greenhouse sensors and classify each reading as OK/HIGH/LOW.
    Input:  current state (may have empty readings)
    Output: state updated with fresh sensor values + sensor_status dict
    """
    print("\n  🔍 [Monitor Agent] Reading sensors...")

    readings = get_sensor_readings()   # simulated sensor data
    scenario = readings.pop("_scenario", "normal")

    # Push sensor values into state
    state["temperature"]   = readings["temperature"]
    state["humidity"]      = readings["humidity"]
    state["soil_moisture"] = readings["soil_moisture"]
    state["light_level"]   = readings["light_level"]
    state["co2_level"]     = readings["co2_level"]

    # Classify each reading
    status = {}
    for sensor, value in readings.items():
        status[sensor] = classify_reading(sensor, value)

    state["sensor_status"] = status

    # Print a clean sensor table
    print(f"     Scenario injected : {scenario}")
    print(f"     ┌{'─'*30}┬{'─'*12}┬{'─'*8}┐")
    print(f"     │ {'Sensor':<28} │ {'Value':>10} │ {'Status':<6} │")
    print(f"     ├{'─'*30}┼{'─'*12}┼{'─'*8}┤")
    sensors_display = [
        ("Temperature",   state["temperature"],   THRESHOLDS["temperature"]["unit"]),
        ("Humidity",      state["humidity"],       THRESHOLDS["humidity"]["unit"]),
        ("Soil Moisture", state["soil_moisture"],  THRESHOLDS["soil_moisture"]["unit"]),
        ("Light Level",   state["light_level"],    THRESHOLDS["light_level"]["unit"]),
        ("CO2 Level",     state["co2_level"],      THRESHOLDS["co2_level"]["unit"]),
    ]
    keys = ["temperature","humidity","soil_moisture","light_level","co2_level"]
    for (label, val, unit), key in zip(sensors_display, keys):
        s = status[key]
        icon = "✅" if s == "OK" else ("🔴" if s == "HIGH" else "🔵")
        print(f"     │ {label:<28} │ {val:>8.1f}{unit[:2]:>2} │ {icon} {s:<4} │")
    print(f"     └{'─'*30}┴{'─'*12}┴{'─'*8}┘")

    return state
