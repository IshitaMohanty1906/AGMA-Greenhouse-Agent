# state.py — Shared State for all agents

from typing import TypedDict, List, Dict, Any

class GreenhouseState(TypedDict):
    # ── Sensor readings ──────────────────────────────
    temperature:    float       # °C
    humidity:       float       # %
    soil_moisture:  float       # %
    light_level:    float       # lux
    co2_level:      float       # ppm

    # ── Agent outputs ────────────────────────────────
    sensor_status:  Dict[str, str]   # "OK" / "HIGH" / "LOW" per sensor
    actions_taken:  List[str]        # decisions by Control Agent
    alerts:         List[str]        # warnings by Alert Agent
    report:         str              # final summary by Report Agent

    # ── Meta ─────────────────────────────────────────
    cycle_count:    int
    crop_type:      str
    error_log:      List[str]
