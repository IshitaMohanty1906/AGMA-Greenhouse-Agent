# agents/control_agent.py — Uses Claude LLM to decide actions, then executes them

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from state import GreenhouseState
from claude_client import call_claude
from tools.actuator_tools import ACTION_MAP
from config import THRESHOLDS, CROP_PROFILES, ACTIVE_CROP


SYSTEM_PROMPT = """You are an expert greenhouse automation controller.
Your job is to decide which actuator actions to take based on real-time sensor readings.

Available actions (use exact names only):
- activate_fan          → when temperature is HIGH or CO2 is HIGH
- activate_irrigation   → when soil_moisture is LOW
- open_vent             → when humidity is HIGH or CO2 is HIGH
- turn_on_grow_lights   → when light_level is LOW
- activate_co2_injector → when co2_level is LOW
- activate_heater       → when temperature is LOW
- no_action             → when all readings are within acceptable range

Rules:
1. Respond ONLY with a comma-separated list of action names. Nothing else.
2. Do not explain. Do not use bullet points. Do not add extra text.
3. If multiple actions are needed, list them all separated by commas.
4. If all is fine, respond with exactly: no_action

Example responses:
activate_fan,open_vent
activate_irrigation
no_action
"""


def control_agent(state: GreenhouseState) -> GreenhouseState:
    """
    CONTROL AGENT
    ─────────────
    Role: Ask Claude LLM to analyze sensor readings and decide what actions to take.
          Then execute those actions via actuator tools.
    Input:  state with sensor readings + sensor_status
    Output: state with actions_taken list filled
    """
    print("\n  🎛️  [Control Agent] Asking Claude for decisions...")

    crop = CROP_PROFILES.get(state.get("crop_type", ACTIVE_CROP), {})

    user_message = f"""
Current greenhouse sensor readings:
- Temperature:   {state['temperature']}°C   → Status: {state['sensor_status'].get('temperature', 'UNKNOWN')}   | Threshold: {THRESHOLDS['temperature']['min']}–{THRESHOLDS['temperature']['max']}°C
- Humidity:      {state['humidity']}%        → Status: {state['sensor_status'].get('humidity', 'UNKNOWN')}   | Threshold: {THRESHOLDS['humidity']['min']}–{THRESHOLDS['humidity']['max']}%
- Soil Moisture: {state['soil_moisture']}%   → Status: {state['sensor_status'].get('soil_moisture', 'UNKNOWN')}   | Threshold: {THRESHOLDS['soil_moisture']['min']}–{THRESHOLDS['soil_moisture']['max']}%
- Light Level:   {state['light_level']} lux → Status: {state['sensor_status'].get('light_level', 'UNKNOWN')}   | Threshold: {THRESHOLDS['light_level']['min']}–{THRESHOLDS['light_level']['max']} lux
- CO2 Level:     {state['co2_level']} ppm   → Status: {state['sensor_status'].get('co2_level', 'UNKNOWN')}   | Threshold: {THRESHOLDS['co2_level']['min']}–{THRESHOLDS['co2_level']['max']} ppm

Active Crop: {state.get('crop_type', ACTIVE_CROP)}
Ideal temperature for crop: {crop.get('temp_ideal', 'N/A')}°C
Ideal humidity for crop: {crop.get('humidity_ideal', 'N/A')}%

Decide which actions to take.
"""

    raw_response = call_claude(SYSTEM_PROMPT, user_message)
    print(f"     Claude decision raw: '{raw_response}'")

    # Parse actions from Claude's response
    actions = [a.strip().lower() for a in raw_response.split(",") if a.strip()]
    valid_actions = [a for a in actions if a in ACTION_MAP]

    if not valid_actions:
        valid_actions = ["no_action"]

    # Execute each action
    executed = []
    for action in valid_actions:
        result = ACTION_MAP[action]()
        executed.append(result)
        print(f"     ⚡ Executed: {result}")

    state["actions_taken"] = executed
    return state
