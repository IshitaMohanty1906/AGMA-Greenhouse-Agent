"""
================================================================
  TOOLS — All 8 Greenhouse Control Tools
  These are the functions the LLM agent can CALL.
================================================================
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from langchain_core.tools import tool

_df = None

def _load_df():
    global _df
    if _df is None:
        _df = pd.read_csv("data/greenhouse_climate_dataset.csv")
        _df['timestamp'] = pd.to_datetime(_df['timestamp'])
    return _df


# ── TOOL 1 ───────────────────────────────────────────────────────
@tool
def get_sensor_data() -> str:
    """
    Read current greenhouse sensor data including temperature,
    humidity, CO2 concentration, light intensity, and soil moisture.
    Always call this tool FIRST before any analysis or decision.
    """
    df = _load_df()
    row = df.sample(1).iloc[0]
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "indoor_temperature_c": round(float(row['indoor_temperature_c']), 2),
        "outdoor_temperature_c": round(float(row['outdoor_temperature_c']), 2),
        "indoor_humidity_pct": round(float(row['indoor_humidity_pct']), 2),
        "co2_ppm": round(float(row['co2_ppm']), 1),
        "light_intensity_lux": round(float(row['light_intensity_lux']), 1),
        "soil_moisture_pct": round(float(row['soil_moisture_pct']), 2),
        "current_ventilation_on": int(row['ventilation_status']),
        "current_heating_on": int(row['heating_status']),
        "current_irrigation_on": int(row['irrigation_status']),
        "current_lighting_on": int(row['supplemental_lighting']),
    }
    return json.dumps(data, indent=2)


# ── TOOL 2 ───────────────────────────────────────────────────────
@tool
def get_historical_data(hours: int = 6) -> str:
    """
    Get historical average sensor readings from past N hours.
    Use to detect trends: is temperature rising or falling over time?
    Args:
        hours: number of past hours to look back (default 6)
    """
    df = _load_df()
    n_rows = min(hours * 12, len(df))
    recent = df.tail(n_rows)
    summary = {
        "period_hours": hours,
        "avg_temperature_c": round(float(recent['indoor_temperature_c'].mean()), 2),
        "max_temperature_c": round(float(recent['indoor_temperature_c'].max()), 2),
        "min_temperature_c": round(float(recent['indoor_temperature_c'].min()), 2),
        "avg_humidity_pct": round(float(recent['indoor_humidity_pct'].mean()), 2),
        "max_humidity_pct": round(float(recent['indoor_humidity_pct'].max()), 2),
        "avg_co2_ppm": round(float(recent['co2_ppm'].mean()), 1),
        "max_co2_ppm": round(float(recent['co2_ppm'].max()), 1),
        "avg_light_lux": round(float(recent['light_intensity_lux'].mean()), 1),
        "avg_soil_moisture_pct": round(float(recent['soil_moisture_pct'].mean()), 2),
        "ventilation_active_pct": round(float(recent['ventilation_status'].mean()) * 100, 1),
        "heating_active_pct": round(float(recent['heating_status'].mean()) * 100, 1),
    }
    return json.dumps(summary, indent=2)


# ── TOOL 3 ───────────────────────────────────────────────────────
@tool
def analyze_conditions(sensor_json: str) -> str:
    """
    Analyze current sensor readings against safe greenhouse thresholds.
    Returns status (NORMAL / WARNING / CRITICAL) for each parameter.
    Also detects compound disease risks.
    Args:
        sensor_json: the JSON string returned by get_sensor_data()
    """
    try:
        data = json.loads(sensor_json)
    except Exception:
        return json.dumps({"error": "Invalid sensor JSON"})

    temp     = data.get("indoor_temperature_c", 25)
    humidity = data.get("indoor_humidity_pct", 60)
    co2      = data.get("co2_ppm", 800)
    light    = data.get("light_intensity_lux", 400)
    soil     = data.get("soil_moisture_pct", 50)
    hour     = datetime.now().hour
    daytime  = 6 <= hour <= 20

    def classify(v, lc, lw, hw, hc):
        if v <= lc or v >= hc: return "CRITICAL"
        if v <= lw or v >= hw: return "WARNING"
        return "NORMAL"

    results = {
        "temperature": {
            "value": temp, "unit": "C",
            "status": classify(temp, 12, 18, 30, 38),
            "normal_range": "18-28C"
        },
        "humidity": {
            "value": humidity, "unit": "%",
            "status": classify(humidity, 35, 50, 80, 90),
            "normal_range": "50-80%"
        },
        "co2": {
            "value": co2, "unit": "ppm",
            "status": classify(co2, 350, 400, 1200, 1600),
            "normal_range": "400-1200 ppm"
        },
        "light": {
            "value": light, "unit": "lux",
            "status": "WARNING" if (daytime and light < 200) else "NORMAL",
            "normal_range": "200-800 lux"
        },
        "soil_moisture": {
            "value": soil, "unit": "%",
            "status": classify(soil, 20, 35, 80, 90),
            "normal_range": "35-75%"
        },
        "compound_risks": [],
        "hour": hour,
        "is_daytime": daytime
    }

    # Compound risk detection
    if temp > 30 and humidity > 80:
        results["compound_risks"].append("FUSARIUM_WILT_RISK: High temp + High humidity")
    if temp < 18 and humidity > 85:
        results["compound_risks"].append("BOTRYTIS_RISK: Low temp + High humidity")
    if daytime and light < 150 and humidity > 70:
        results["compound_risks"].append("POWDERY_MILDEW_RISK: Low light + High humidity")
    if soil > 85:
        results["compound_risks"].append("ROOT_ROT_RISK: Soil waterlogged")

    statuses = [results[k]["status"] for k in ["temperature","humidity","co2","light","soil_moisture"]]
    if "CRITICAL" in statuses or results["compound_risks"]:
        results["overall_severity"] = "CRITICAL"
    elif "WARNING" in statuses:
        results["overall_severity"] = "WARNING"
    else:
        results["overall_severity"] = "NORMAL"

    return json.dumps(results, indent=2)


# ── TOOL 4 ───────────────────────────────────────────────────────
@tool
def control_ventilation(action: str, level: str = "full") -> str:
    """
    Control the greenhouse ventilation / air circulation system.
    Args:
        action: "ON" to activate ventilation, "OFF" to stop
        level: "full" for 100%, "half" for 50%, "low" for 30%
    Use when: temperature too high, CO2 too high, or humidity too high.
    Do NOT use when temperature is already critically low.
    """
    pcts = {"full": 100, "half": 50, "low": 30}
    pct  = pcts.get(level.lower(), 100)
    if action.upper() == "ON":
        result = {"system": "Ventilation", "status": "ACTIVATED",
                  "level": f"{pct}%",
                  "effect": "Temperature and CO2 will decrease in 10-15 minutes"}
    else:
        result = {"system": "Ventilation", "status": "DEACTIVATED",
                  "effect": "Temperature will stabilize"}
    return json.dumps(result)


# ── TOOL 5 ───────────────────────────────────────────────────────
@tool
def control_irrigation(action: str, duration_minutes: int = 10) -> str:
    """
    Control the greenhouse drip irrigation system.
    Args:
        action: "ON" to start watering, "OFF" to stop
        duration_minutes: how long to irrigate in minutes (default 10)
    Use when: soil moisture below 35%.
    Do NOT use when: humidity above 80% (worsens fungal risk).
    Do NOT use when: soil moisture already above 75%.
    """
    if action.upper() == "ON":
        result = {"system": "Irrigation", "status": "ACTIVATED",
                  "duration": f"{duration_minutes} minutes",
                  "effect": "Soil moisture will increase 10-15% per cycle"}
    else:
        result = {"system": "Irrigation", "status": "DEACTIVATED",
                  "effect": "Soil moisture will slowly decrease"}
    return json.dumps(result)


# ── TOOL 6 ───────────────────────────────────────────────────────
@tool
def control_heating(action: str, target_temp_c: float = 20.0) -> str:
    """
    Control the greenhouse heating system.
    Args:
        action: "ON" to activate heating, "OFF" to deactivate
        target_temp_c: desired temperature in Celsius (default 20.0)
    Use when: temperature falls below 18 degrees Celsius.
    Do NOT run simultaneously with maximum ventilation.
    """
    if action.upper() == "ON":
        result = {"system": "Heating", "status": "ACTIVATED",
                  "target": f"{target_temp_c}C",
                  "effect": "Temperature will rise over 20-30 minutes"}
    else:
        result = {"system": "Heating", "status": "DEACTIVATED",
                  "effect": "Temperature will slowly decrease"}
    return json.dumps(result)


# ── TOOL 7 ───────────────────────────────────────────────────────
@tool
def control_lighting(action: str, intensity: str = "full") -> str:
    """
    Control the supplemental grow lighting system.
    Args:
        action: "ON" to activate lights, "OFF" to turn off
        intensity: "full" for 100%, "half" for 50%
    Use when: light intensity below 200 lux during daytime (6am-8pm).
    Do NOT activate during night hours (10pm-6am). Plants need dark period.
    """
    pcts = {"full": 100, "half": 50}
    pct  = pcts.get(intensity.lower(), 100)
    if action.upper() == "ON":
        result = {"system": "Grow Lights", "status": "ACTIVATED",
                  "intensity": f"{pct}%",
                  "effect": "Light levels increase immediately"}
    else:
        result = {"system": "Grow Lights", "status": "DEACTIVATED"}
    return json.dumps(result)


# ── TOOL 8 ───────────────────────────────────────────────────────
@tool
def send_alert(severity: str, condition: str, message: str, action_taken: str) -> str:
    """
    Send a greenhouse alert notification.
    Args:
        severity: one of INFO, WARNING, CRITICAL, EMERGENCY
        condition: short label like "High Temperature" or "Fusarium Risk"
        message: detailed description of the problem
        action_taken: what the agent already did or recommends
    Call this for any WARNING, CRITICAL, or EMERGENCY condition.
    """
    icons = {"INFO": "i", "WARNING": "!", "CRITICAL": "!!", "EMERGENCY": "SOS"}
    icon  = icons.get(severity.upper(), "!")
    alert = {
        "alert_id": f"GH-{datetime.now().strftime('%H%M%S')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "severity": severity.upper(),
        "icon": icon,
        "condition": condition,
        "message": message,
        "action_taken": action_taken,
        "status": "SENT"
    }
    print(f"\n  [{icon}] ALERT - {severity.upper()}: {condition}")
    print(f"      {message}")
    print(f"      Action: {action_taken}\n")
    return json.dumps(alert)


# ── ALL TOOLS LIST ────────────────────────────────────────────────
ALL_TOOLS = [
    get_sensor_data,
    get_historical_data,
    analyze_conditions,
    control_ventilation,
    control_irrigation,
    control_heating,
    control_lighting,
    send_alert,
]
