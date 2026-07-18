# agents/alert_agent.py — Detects critical conditions and raises alerts

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from state import GreenhouseState
from tools.actuator_tools import send_sms_alert
from config import THRESHOLDS


# Critical thresholds (beyond normal HIGH/LOW — plant survival danger zone)
CRITICAL = {
    "temperature":   {"max": 38.0, "min": 10.0},
    "humidity":      {"max": 90.0, "min": 20.0},
    "soil_moisture": {"max": 90.0, "min": 15.0},
    "co2_level":     {"max": 1400.0},
}


def alert_agent(state: GreenhouseState) -> GreenhouseState:
    """
    ALERT AGENT
    ───────────
    Role: Check sensor readings against CRITICAL thresholds.
          Generate alerts and simulate farmer notification.
    Input:  state with sensor readings
    Output: state with alerts list filled
    """
    print("\n  🚨 [Alert Agent] Checking for critical conditions...")

    alerts = []

    # Temperature checks
    temp = state["temperature"]
    if temp > CRITICAL["temperature"]["max"]:
        msg = f"🔴 CRITICAL: Temperature {temp}°C is dangerously HIGH! Plants at risk!"
        alerts.append(msg)
        send_sms_alert(msg)
    elif temp < CRITICAL["temperature"]["min"]:
        msg = f"🔵 CRITICAL: Temperature {temp}°C is dangerously LOW! Frost risk!"
        alerts.append(msg)
        send_sms_alert(msg)

    # Humidity checks
    hum = state["humidity"]
    if hum > CRITICAL["humidity"]["max"]:
        msg = f"🔴 CRITICAL: Humidity {hum}% is too HIGH! Fungal disease risk!"
        alerts.append(msg)
        send_sms_alert(msg)
    elif hum < CRITICAL["humidity"]["min"]:
        msg = f"🔵 CRITICAL: Humidity {hum}% is too LOW! Plants dehydrating!"
        alerts.append(msg)
        send_sms_alert(msg)

    # Soil moisture checks
    soil = state["soil_moisture"]
    if soil > CRITICAL["soil_moisture"]["max"]:
        msg = f"🔴 CRITICAL: Soil moisture {soil}% — root rot danger!"
        alerts.append(msg)
        send_sms_alert(msg)
    elif soil < CRITICAL["soil_moisture"]["min"]:
        msg = f"🔵 CRITICAL: Soil moisture {soil}% — severe drought stress!"
        alerts.append(msg)
        send_sms_alert(msg)

    # CO2 checks
    co2 = state["co2_level"]
    if co2 > CRITICAL["co2_level"]["max"]:
        msg = f"🔴 CRITICAL: CO2 {co2} ppm — toxic levels! Open vents immediately!"
        alerts.append(msg)
        send_sms_alert(msg)

    # Warn level (not critical but outside normal range)
    status = state.get("sensor_status", {})
    for sensor, st in status.items():
        if st in ("HIGH", "LOW") and sensor not in []:
            warn_msg = f"⚠️  WARNING: {sensor.replace('_',' ').title()} is {st}"
            if warn_msg not in alerts:     # avoid duplicate with critical
                alerts.append(warn_msg)

    if not alerts:
        alerts.append("✅ All parameters within safe range. No alerts.")
        print("     ✅ No critical alerts.")
    else:
        for a in alerts:
            print(f"     {a}")

    state["alerts"] = alerts
    return state
