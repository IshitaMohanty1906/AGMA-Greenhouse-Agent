"""
step3_main_agent.py
====================
AGMA — Automated Greenhouse Manager Agent
LangGraph Multi-Agent System with Ollama (offline LLM)

Agents: SensorAgent → AnalysisAgent → DecisionAgent → AlertAgent
LLM:    Ollama local (llama3.2:3b) — no API key needed
RAG:    ChromaDB + MiniLM-L6-v2 embeddings
"""

import os
import re
import json
import time
from typing import TypedDict, List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ── RAG Setup ────────────────────────────────────────────────────────────────
RAG_AVAILABLE = False
_vectorstore  = None

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    _embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    _chroma     = chromadb.PersistentClient(path="rag/chroma_db")
    RAG_AVAILABLE = True
    print("[AGMA] ChromaDB RAG loaded successfully.")
except Exception as e:
    print(f"[AGMA] RAG not available: {e}")


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant passages from ChromaDB."""
    if not RAG_AVAILABLE:
        return ""
    try:
        collections = _chroma.list_collections()
        passages = []
        q_emb = _embedder.encode([query])[0].tolist()
        for col in collections:
            c = _chroma.get_collection(col.name)
            results = c.query(
                query_embeddings=[q_emb],
                n_results=min(k, c.count())
            )
            if results and results["documents"]:
                passages.extend(results["documents"][0])
        return "\n".join(passages[:k])
    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return ""


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm():
    """Return local Ollama LLM instance."""
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0.1
    )


# ── State ─────────────────────────────────────────────────────────────────────

class GreenhouseState(TypedDict):
    raw_sensor   : Dict[str, Any]
    current_data : Dict[str, Any]
    historical   : List[Dict]
    rag_context  : str
    analysis     : str
    decisions    : Dict[str, int]
    explanation  : str
    alerts       : List[str]


# ── Agent Config ──────────────────────────────────────────────────────────────

class AgentConfig:
    def __init__(self,
                 use_rag=True,
                 use_history=True,
                 multi_agent=True):
        self.use_rag     = use_rag
        self.use_history = use_history
        self.multi_agent = multi_agent

DEFAULT_CONFIG = AgentConfig(use_rag=True,
                             use_history=True,
                             multi_agent=True)


# ── Historical store (in-memory) ──────────────────────────────────────────────
_history_buffer: List[Dict] = []

def get_historical_data(window: int = 12) -> List[Dict]:
    return _history_buffer[-window:]

def push_to_history(row: Dict):
    _history_buffer.append(row)
    if len(_history_buffer) > 100:
        _history_buffer.pop(0)


# ── Sensor Agent ──────────────────────────────────────────────────────────────

def sensor_agent(state: GreenhouseState,
                 config: AgentConfig = DEFAULT_CONFIG) -> GreenhouseState:
    """
    Phase 1: Load, validate, and flag sensor readings.
    """
    row = state["raw_sensor"]

    current = {
        "indoor_temperature_c"  : float(row.get("indoor_temperature_c",   row.get("indoor_temperature",   25.0))),
        "outdoor_temperature_c" : float(row.get("outdoor_temperature_c",  row.get("outdoor_temperature",  20.0))),
        "indoor_humidity_pct"   : float(row.get("indoor_humidity_pct",    row.get("indoor_humidity",      60.0))),
        "outdoor_humidity_pct"  : float(row.get("outdoor_humidity_pct",   row.get("outdoor_humidity",     55.0))),
        "co2_ppm"               : float(row.get("co2_ppm",                row.get("co2_concentration",    800.0))),
        "light_intensity_lux"   : float(row.get("light_intensity_lux",   row.get("light_intensity",      200.0))),
        "soil_moisture_pct"     : float(row.get("soil_moisture_pct",      row.get("soil_moisture",        50.0))),
        "energy_consumption_kwh": float(row.get("energy_consumption_kwh", row.get("energy_consumption",   0.0))),
        "water_consumption_L"   : float(row.get("water_consumption_L",    row.get("water_consumption",    0.0))),
    }

    historical = get_historical_data(12) if config.use_history else []
    push_to_history(current)

    state["current_data"] = current
    state["historical"]   = historical
    return state


# ── Analysis Agent ────────────────────────────────────────────────────────────

def analysis_agent(state: GreenhouseState,
                   config: AgentConfig = DEFAULT_CONFIG) -> GreenhouseState:
    """
    Phase 2: RAG retrieval + condition analysis.
    """
    cd  = state["current_data"]
    rag = ""

    if config.use_rag and RAG_AVAILABLE:
        query = (
            f"temperature {cd['indoor_temperature_c']} "
            f"humidity {cd['indoor_humidity_pct']} "
            f"co2 {cd['co2_ppm']} "
            f"soil {cd['soil_moisture_pct']}"
        )
        rag = retrieve_context(query, k=3)

    # Condition analysis
    flags = []
    if cd["indoor_temperature_c"] > 28:
        flags.append("CRITICAL: temperature above 28C")
    if cd["indoor_temperature_c"] < 18:
        flags.append("CRITICAL: temperature below 18C")
    if cd["indoor_humidity_pct"] > 85:
        flags.append("WARNING: humidity above 85%")
    if cd["co2_ppm"] > 1200:
        flags.append("WARNING: CO2 above 1200 ppm")
    if cd["soil_moisture_pct"] < 35:
        flags.append("WARNING: soil moisture below 35%")
    if cd["light_intensity_lux"] < 150:
        flags.append("INFO: light below 150 lux")

    # Compound risk
    if cd["indoor_temperature_c"] > 30 and cd["indoor_humidity_pct"] > 80:
        flags.append("COMPOUND RISK: Botrytis disease conditions detected")

    analysis = (
        f"Condition flags: {'; '.join(flags) if flags else 'All parameters normal'}. "
        f"Temp trend: {'rising' if len(state['historical']) > 1 and cd['indoor_temperature_c'] > state['historical'][-1].get('indoor_temperature_c', cd['indoor_temperature_c']) else 'stable/falling'}."
    )

    state["rag_context"] = rag
    state["analysis"]    = analysis
    return state


# ── Decision Agent ────────────────────────────────────────────────────────────

def decision_agent(state: GreenhouseState) -> GreenhouseState:
    """
    Phase 3: ReAct-style LLM decision making with tool execution.
    """
    llm = get_llm()
    cd  = state["current_data"]
    rag = state["rag_context"]
    ana = state["analysis"]

    prompt = f"""You are an autonomous greenhouse climate controller.

Current sensor readings:
- Indoor Temperature : {cd.get('indoor_temperature_c', 'N/A')} C
- Outdoor Temperature: {cd.get('outdoor_temperature_c', 'N/A')} C
- Indoor Humidity    : {cd.get('indoor_humidity_pct', 'N/A')} %
- Outdoor Humidity   : {cd.get('outdoor_humidity_pct', 'N/A')} %
- CO2                : {cd.get('co2_ppm', 'N/A')} ppm
- Light Intensity    : {cd.get('light_intensity_lux', 'N/A')} lux
- Soil Moisture      : {cd.get('soil_moisture_pct', 'N/A')} %

System Analysis: {ana}

{f'Domain Knowledge:{chr(10)}{rag}' if rag else ''}

Decision rules:
- Ventilate if CO2 > 1200 OR humidity > 85% OR temperature > 28C
- Heat if temperature < 18C
- Irrigate if soil moisture < 35%
- Illuminate if light < 150 lux

Respond ONLY in this exact JSON format, nothing else:
{{
    "ventilation_status": 0,
    "heating_status": 0,
    "irrigation_status": 0,
    "supplemental_lighting": 0,
    "explanation": "one sentence reason for decisions"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content  = response.content
        match    = re.search(r'\{.*\}', content, re.DOTALL)

        if match:
            result = json.loads(match.group())
            state["decisions"] = {
                "ventilation": int(result.get("ventilation_status", 0)),
                "heating"    : int(result.get("heating_status",     0)),
                "irrigation" : int(result.get("irrigation_status",  0)),
                "lighting"   : int(result.get("supplemental_lighting", 0)),
            }
            state["explanation"] = result.get("explanation", "")
        else:
            print(f"[DecisionAgent] JSON parse failed. Raw: {content[:200]}")
            state["decisions"]   = _fallback_decisions(cd)
            state["explanation"] = "Fallback threshold-based decision"

    except Exception as e:
        print(f"[DecisionAgent] Error: {e}")
        state["decisions"]   = _fallback_decisions(cd)
        state["explanation"] = f"Error fallback: {str(e)}"

    return state


def _fallback_decisions(cd: Dict) -> Dict[str, int]:
    """Rule-based fallback if LLM fails."""
    return {
        "ventilation": 1 if (cd.get("co2_ppm", 0) > 1200 or
                             cd.get("indoor_humidity_pct", 0) > 85 or
                             cd.get("indoor_temperature_c", 0) > 28) else 0,
        "heating"    : 1 if cd.get("indoor_temperature_c", 25) < 18 else 0,
        "irrigation" : 1 if cd.get("soil_moisture_pct", 50) < 35 else 0,
        "lighting"   : 1 if cd.get("light_intensity_lux", 200) < 150 else 0,
    }


# ── Alert Agent ───────────────────────────────────────────────────────────────

def alert_agent(state: GreenhouseState) -> GreenhouseState:
    """
    Phase 4: Check for critical conditions and dispatch alerts.
    """
    cd     = state["current_data"]
    alerts = []

    if cd.get("indoor_temperature_c", 0) > 35:
        alerts.append("CRITICAL: Temperature exceeds 35C — pollen sterility risk")

    if cd.get("indoor_temperature_c", 25) < 10:
        alerts.append("CRITICAL: Temperature below 10C — frost risk")

    if cd.get("indoor_humidity_pct", 0) > 90:
        alerts.append("CRITICAL: Humidity above 90% — severe disease risk")

    if cd.get("co2_ppm", 0) > 1500:
        alerts.append("CRITICAL: CO2 above 1500 ppm — dangerous levels")

    if (cd.get("indoor_temperature_c", 0) > 30 and
            cd.get("indoor_humidity_pct", 0) > 80):
        alerts.append("WARNING: Compound Botrytis risk — high temp + high humidity")

    d = state.get("decisions", {})
    if d.get("heating") == 1 and d.get("ventilation") == 1:
        alerts.append("INFO: Simultaneous heating and ventilation — check efficiency")

    state["alerts"] = alerts
    return state


# ── Single Combined Agent (ablation variant) ──────────────────────────────────

def single_combined_agent(state: GreenhouseState) -> GreenhouseState:
    """
    Ablation variant: all reasoning in one LLM call.
    Used when config.multi_agent = False.
    """
    llm = get_llm()

    prompt = f"""You are a greenhouse climate manager.

Current sensor readings:
{state['current_data']}

Historical readings (last 12):
{state['historical']}

RAG Context:
{state['rag_context']}

Make binary control decisions (0 or 1) for:
- ventilation
- irrigation
- heating
- lighting

Respond ONLY in this exact JSON format:
{{
    "ventilation": 0,
    "irrigation": 0,
    "heating": 0,
    "lighting": 0,
    "explanation": "brief reason"
}}"""

    try:
        response  = llm.invoke([HumanMessage(content=prompt)])
        content   = response.content
        match     = re.search(r'\{.*\}', content, re.DOTALL)

        if match:
            result = json.loads(match.group())
            state["decisions"] = {
                "ventilation": int(result.get("ventilation", 0)),
                "heating"    : int(result.get("heating",     0)),
                "irrigation" : int(result.get("irrigation",  0)),
                "lighting"   : int(result.get("lighting",    0)),
            }
            state["explanation"] = result.get("explanation", "")
        else:
            state["decisions"]   = _fallback_decisions(state["current_data"])
            state["explanation"] = "parse error — fallback used"

    except Exception as e:
        print(f"[SingleAgent] Error: {e}")
        state["decisions"]   = _fallback_decisions(state["current_data"])
        state["explanation"] = str(e)

    return state


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(config: AgentConfig = DEFAULT_CONFIG):
    """Build and compile LangGraph StateGraph."""
    graph = StateGraph(GreenhouseState)

    if config.multi_agent:
        graph.add_node("sensor",   lambda s: sensor_agent(s, config))
        graph.add_node("analysis", lambda s: analysis_agent(s, config))
        graph.add_node("decision", decision_agent)
        graph.add_node("alert",    alert_agent)

        graph.set_entry_point("sensor")
        graph.add_edge("sensor",   "analysis")
        graph.add_edge("analysis", "decision")
        graph.add_edge("decision", "alert")
        graph.add_edge("alert",    END)
    else:
        graph.add_node("single", single_combined_agent)
        graph.set_entry_point("single")
        graph.add_edge("single", END)

    return graph.compile()


# ── Run Agent ─────────────────────────────────────────────────────────────────

def run_agent(sensor_row: dict,
              config: AgentConfig = DEFAULT_CONFIG) -> dict:
    """
    Run complete agent pipeline for one sensor reading.
    Returns decisions, explanation, alerts, response time.
    """
    graph = build_graph(config)

    initial_state: GreenhouseState = {
        "raw_sensor"  : sensor_row,
        "current_data": {},
        "historical"  : [],
        "rag_context" : "",
        "analysis"    : "",
        "decisions"   : {},
        "explanation" : "",
        "alerts"      : [],
    }

    start      = time.perf_counter()
    final      = graph.invoke(initial_state)
    elapsed    = time.perf_counter() - start

    return {
        "decisions"    : final["decisions"],
        "explanation"  : final["explanation"],
        "alerts"       : final["alerts"],
        "response_time": round(elapsed, 3),
    }


# ── Eval helper ───────────────────────────────────────────────────────────────

def run_agent_for_eval(sensor_row: dict,
                       config=None) -> dict:
    """
    Evaluation wrapper — same as run_agent but config-aware.
    Used by final_evaluation.py and ablation_study.py.
    """
    if config is None:
        config = DEFAULT_CONFIG
    return run_agent(sensor_row, config)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("AGMA — Automated Greenhouse Manager Agent")
    print("LLM: Ollama llama3.2:3b (offline)")
    print("="*55)

    # Test with one sample row
    test_row = {
        "indoor_temperature_c"  : 34.2,
        "outdoor_temperature_c" : 24.1,
        "indoor_humidity_pct"   : 78.3,
        "outdoor_humidity_pct"  : 60.0,
        "co2_ppm"               : 1182.0,
        "light_intensity_lux"   : 274.1,
        "soil_moisture_pct"     : 30.1,
        "energy_consumption_kwh": 2.4,
        "water_consumption_L"   : 12.0,
    }

    print("\nRunning agent on test sensor row...")
    print(f"Input: Temp={test_row['indoor_temperature_c']}C | "
          f"CO2={test_row['co2_ppm']}ppm | "
          f"Humidity={test_row['indoor_humidity_pct']}% | "
          f"Soil={test_row['soil_moisture_pct']}%")

    result = run_agent(test_row)

    print(f"\nDecisions    : {result['decisions']}")
    print(f"Explanation  : {result['explanation']}")
    print(f"Alerts       : {result['alerts']}")
    print(f"Response time: {result['response_time']}s")
    print("\nAGMA running successfully.")