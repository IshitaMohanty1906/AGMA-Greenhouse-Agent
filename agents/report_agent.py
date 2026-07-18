# agents/report_agent.py — Generates a structured report using Claude

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from state import GreenhouseState
from claude_client import call_claude


REPORT_SYSTEM_PROMPT = """You are a greenhouse management reporting assistant.
Given sensor readings, actions taken, and alerts, write a concise professional
cycle report for a farmer. Use plain text (no markdown). Keep it under 150 words.
Structure: 1) Summary sentence. 2) Key issues found. 3) Actions taken. 4) Recommendation.
"""


def report_agent(state: GreenhouseState) -> GreenhouseState:
    """
    REPORT AGENT
    ────────────
    Role: Use Claude to synthesize all agent outputs into a readable report.
    Input:  full state (readings + actions + alerts)
    Output: state with report string
    """
    print("\n  📊 [Report Agent] Generating cycle report...")

    user_message = f"""
Cycle #{state.get('cycle_count', 0) + 1} — Greenhouse Report Data:

SENSOR READINGS:
- Temperature:   {state['temperature']}°C
- Humidity:      {state['humidity']}%
- Soil Moisture: {state['soil_moisture']}%
- Light Level:   {state['light_level']} lux
- CO2 Level:     {state['co2_level']} ppm

SENSOR STATUS: {state.get('sensor_status', {})}

ACTIONS TAKEN:
{chr(10).join(state.get('actions_taken', ['None'])) or 'None'}

ALERTS RAISED:
{chr(10).join(state.get('alerts', ['None'])) or 'None'}

Active Crop: {state.get('crop_type', 'tomato')}

Generate the cycle report now.
"""

    report = call_claude(REPORT_SYSTEM_PROMPT, user_message, max_tokens=300)

    print("\n" + "="*70)
    print("  📋 CYCLE REPORT:")
    print("="*70)
    print(report)
    print("="*70)

    state["report"] = report
    return state
