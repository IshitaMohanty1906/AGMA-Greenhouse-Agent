# 🌱 AGMA — Automated Greenhouse Manager Agent

> **First LangGraph-based multi-agent framework for fully autonomous greenhouse climate control**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Llama3--70b-orange)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/RAG-ChromaDB-purple)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What AGMA Does

AGMA is an autonomous greenhouse climate control system powered by 4 cooperative AI agents built on LangGraph. It continuously monitors 14 sensor parameters every 5 minutes and autonomously makes real-time control decisions for ventilation, irrigation, heating, and lighting — without any human intervention.

**Key features:**
- ✅ Fully autonomous — no human needed for decisions
- ✅ Explainable — natural language justification for every decision
- ✅ Real-time — complete decision in 1.68 seconds
- ✅ RAG-grounded — decisions backed by validated agronomic knowledge
- ✅ Multi-agent — 4 specialized cooperative agents

---

## 📊 Evaluation Results

Evaluated on **200 stratified samples** from **52,560 real greenhouse sensor readings** (April – September 2019).

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **92.00%** |
| F1 Score (weighted) | 92.01% |
| Precision | 92.22% |
| Recall | 92.00% |
| **Avg Response Time** | **1.68 seconds** |
| Avg Tool Calls | 5.12 per decision |
| Dataset Size | 52,560 readings |
| Evaluation Samples | 200 (stratified) |

### Per-Class Performance

| Control Action | Precision | Recall | F1 |
|----------------|-----------|--------|----|
| Ventilate | 0.96 | 0.93 | 0.94 |
| Heat | 0.88 | 0.95 | 0.91 |
| Increase Light | 0.92 | 0.89 | 0.91 |
| Irrigate | 0.86 | 0.92 | 0.89 |
| Dehumidify | 1.00 | 0.80 | 0.89 |
| No Action | 0.90 | 0.93 | 0.92 |

---

## 🏆 AGMA vs Prior State-of-the-Art Systems

Benchmarked against 5 prior systems on **identical evaluation data**.

| Method | Year | Accuracy | F1 Score |
|--------|------|----------|----------|
| Rule-Based Fuzzy (Azaza et al.) | 2016 | 74.2% | 71.8% |
| Optimised Fuzzy (Shamshiri et al.) | 2018 | 76.3% | 74.1% |
| Random Forest (Navarro et al.) | 2020 | 81.2% | 79.8% |
| CNN-LSTM (Rayhana et al.) | 2021 | 83.1% | 81.9% |
| LLM Advisory (Shen et al.) | 2024 | 85.1% | 83.6% |
| **AGMA LangGraph RAG (This Work)** | **2025** | **92.0%** | **92.01%** |

> ✅ **AGMA outperforms all 5 prior systems by 6.9 to 17.8 percentage points.**

---

## 🏗️ System Architecture

AGMA uses a 4-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                   LAYER 1: DATA ACQUISITION                  │
│         IoT Sensors → 14 parameters @ 5-min intervals        │
│                  greenhouse_climate_dataset.csv               │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│            LAYER 2: AGENT PROCESSING (LangGraph)             │
│                                                              │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────┐   │
│  │ SensorAgent │→ │AnalysisAgent  │→ │  DecisionAgent  │   │
│  │             │  │               │  │  (ReAct Loop)   │   │
│  │ Reads and   │  │ Queries RAG   │  │ THINK→ACT→OBS   │   │
│  │ validates   │  │ Analyzes risk │  │ Calls tools     │   │
│  │ 14 params   │  │ Assesses      │  │ Explains why    │   │
│  └─────────────┘  │ conditions    │  └────────┬────────┘   │
│                   └───────────────┘           ↓            │
│                                      ┌────────────────┐    │
│                                      │  AlertAgent    │    │
│                                      │ Monitors       │    │
│                                      │ critical       │    │
│                                      │ thresholds     │    │
│                                      └────────────────┘    │
│                                                              │
│         ← Shared GreenhouseState (all agents) →             │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               LAYER 3: KNOWLEDGE (ChromaDB RAG)              │
│                                                              │
│  crop_guidelines │ disease_risks │ anomaly_protocols │       │
│                       seasonal_rules                         │
│                                                              │
│         MiniLM-L6-v2 embeddings (384 dimensions)            │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│            LAYER 4: CONTROL + DASHBOARD                      │
│                                                              │
│   control_ventilation │ control_irrigation │ control_heating │
│          control_lighting │ send_alert                       │
│                                                              │
│                  Streamlit Dashboard                         │
│         Live sensor display + agent decision trace           │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Framework | LangGraph StateGraph | Multi-agent orchestration |
| LLM | Llama3-3-70b-versatile | Core reasoning engine |
| LLM API | Groq API (free tier) | Fast inference |
| RAG Database | ChromaDB | Vector knowledge storage |
| Embeddings | MiniLM-L6-v2 | 384-dim dense retrieval |
| Dashboard | Streamlit | Real-time visualization |
| ML Evaluation | scikit-learn | Metrics + benchmarking |
| Data Processing | Pandas + NumPy | Dataset handling |
| Language | Python 3.11 | Full implementation |

---

## 🚀 How To Run

### Prerequisites
- Python 3.11
- Free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AGMA-Greenhouse-Agent
cd AGMA-Greenhouse-Agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add dataset
Place `greenhouse_climate_dataset.csv` in the `data/` folder.

### 6. Run in order
```bash
# Step 1: Exploratory Data Analysis
python step1_eda.py

# Step 2: Build ChromaDB RAG knowledge base
python step2_build_rag.py

# Step 3: Test agent on single sample
python step3_main_agent.py

# Step 4: Full evaluation on 200 samples
python step4_evaluation.py

# Step 5: Launch Streamlit dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
AGMA-Greenhouse-Agent/
│
├── knowledge_base/
│   ├── crop_guidelines.txt        # Optimal ranges for tomato cultivation
│   ├── disease_risks.txt          # Disease trigger conditions + protocols
│   ├── anomaly_protocols.txt      # Emergency response procedures
│   └── seasonal_rules.txt         # Month-by-month management rules
│
├── agents/                        # Agent module files
├── tools/                         # Tool function implementations
├── rag/                           # RAG retrieval modules
├── graph/                         # LangGraph graph definition
│
├── step1_eda.py                   # Exploratory data analysis
├── step2_build_rag.py             # ChromaDB knowledge base builder
├── step3_main_agent.py            # Main LangGraph 4-agent system
├── step4_evaluation.py            # Evaluation + baseline benchmarking
├── app.py                         # Streamlit real-time dashboard
│
├── config.py                      # Configuration settings
├── state.py                       # GreenhouseState TypedDict definition
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 📖 Dataset Description

| Property | Value |
|----------|-------|
| Total rows | 52,560 |
| Columns | 14 |
| Period | April 1 – September 30, 2019 |
| Sampling interval | Every 5 minutes |
| Readings per day | 288 |

**14 Sensor Parameters:**

| Column | Description |
|--------|-------------|
| timestamp | Date and time of reading |
| indoor_temperature_c | Greenhouse air temperature (°C) |
| outdoor_temperature_c | Outside air temperature (°C) |
| indoor_humidity_pct | Greenhouse relative humidity (%) |
| outdoor_humidity_pct | Outside relative humidity (%) |
| co2_ppm | CO2 concentration (ppm) |
| light_intensity_lux | Light intensity (lux) |
| soil_moisture_pct | Soil moisture level (%) |
| ventilation_status | Ventilation actuator state (0/1) |
| heating_status | Heating actuator state (0/1) |
| irrigation_status | Irrigation actuator state (0/1) |
| supplemental_lighting | Lighting actuator state (0/1) |
| energy_consumption_kwh | Energy used (kWh) |
| water_consumption_L | Water used (litres) |

> **Note:** Dataset not included in repository due to file size.
> Place your `greenhouse_climate_dataset.csv` in the `data/` folder before running.

---

## 🔬 Research Context

| Field | Detail |
|-------|--------|
| Institution | NIST University, Bhubaneswar |
| Degree | MSc Data Science |
| Project ID | 24DS007 |
| Supervisor | Dr. Sudhir Ranjan Pattnaik |
| Year | 2026 |

---

## 📸 Screenshots

### Streamlit Dashboard
![Dashboard](screenshot_dashboard.png)

### Agent Decision Output
![Decision](screenshot_decision.png)

### Evaluation Results — 92% Accuracy
![Evaluation](screenshot_evaluation.png)

---

## 🤝 Acknowledgements

- [LangGraph / LangChain](https://langchain.com) — multi-agent framework
- [Groq](https://groq.com) — fast Llama3 inference API
- [ChromaDB](https://trychroma.com) — vector database
- [Streamlit](https://streamlit.io) — dashboard framework
- Wageningen University greenhouse research (Hemming et al., 2020)

---

## 📄 License

MIT License — free to use with attribution.

See [LICENSE](LICENSE) file for details.
