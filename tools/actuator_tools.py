# tools/actuator_tools.py
# Simulated actuators — in a real system these would send GPIO/MQTT signals

import time

ACTUATOR_LOG = []   # keeps history of all actuator events


def activate_fan(speed: str = "medium") -> str:
    msg = f"[FAN] Activated at {speed} speed — reducing temperature & improving airflow."
    ACTUATOR_LOG.append(msg)
    return msg

def deactivate_fan() -> str:
    msg = "[FAN] Deactivated."
    ACTUATOR_LOG.append(msg)
    return msg

def activate_irrigation(duration_sec: int = 30) -> str:
    msg = f"[IRRIGATION] Started for {duration_sec}s — increasing soil moisture."
    ACTUATOR_LOG.append(msg)
    return msg

def stop_irrigation() -> str:
    msg = "[IRRIGATION] Stopped."
    ACTUATOR_LOG.append(msg)
    return msg

def open_vent(percentage: int = 50) -> str:
    msg = f"[VENT] Opened {percentage}% — adjusting humidity & CO2."
    ACTUATOR_LOG.append(msg)
    return msg

def close_vent() -> str:
    msg = "[VENT] Closed."
    ACTUATOR_LOG.append(msg)
    return msg

def turn_on_grow_lights(intensity: str = "full") -> str:
    msg = f"[LIGHTS] Grow lights ON at {intensity} intensity — supplementing light."
    ACTUATOR_LOG.append(msg)
    return msg

def turn_off_grow_lights() -> str:
    msg = "[LIGHTS] Grow lights OFF."
    ACTUATOR_LOG.append(msg)
    return msg

def activate_co2_injector() -> str:
    msg = "[CO2 INJECTOR] CO2 supplementation started."
    ACTUATOR_LOG.append(msg)
    return msg

def activate_heater() -> str:
    msg = "[HEATER] Heater activated — raising temperature."
    ACTUATOR_LOG.append(msg)
    return msg

def send_sms_alert(message: str) -> str:
    msg = f"[SMS ALERT] → Farmer notified: {message}"
    ACTUATOR_LOG.append(msg)
    return msg

# Map action string → function (used by Control Agent to execute decisions)
ACTION_MAP = {
    "activate_fan":         activate_fan,
    "activate_irrigation":  activate_irrigation,
    "open_vent":            open_vent,
    "turn_on_grow_lights":  turn_on_grow_lights,
    "activate_co2_injector":activate_co2_injector,
    "activate_heater":      activate_heater,
    "no_action":            lambda: "[NO ACTION] Conditions are optimal. No intervention needed.",
}
