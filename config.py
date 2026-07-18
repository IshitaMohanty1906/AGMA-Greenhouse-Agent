# config.py — Thresholds & Settings

THRESHOLDS = {
    "temperature": {"min": 18.0, "max": 35.0, "unit": "°C"},
    "humidity":    {"min": 40.0, "max": 80.0, "unit": "%"},
    "soil_moisture": {"min": 30.0, "max": 75.0, "unit": "%"},
    "light_level": {"min": 300.0, "max": 900.0, "unit": "lux"},
    "co2_level":   {"min": 350.0, "max": 1200.0, "unit": "ppm"},
}

CROP_PROFILES = {
    "tomato":   {"temp_ideal": 24, "humidity_ideal": 65, "light_ideal": 700},
    "lettuce":  {"temp_ideal": 18, "humidity_ideal": 70, "light_ideal": 400},
    "cucumber": {"temp_ideal": 26, "humidity_ideal": 75, "light_ideal": 650},
}

ACTIVE_CROP = "tomato"
MONITORING_CYCLES = 3       # how many cycles to run in main.py
MODEL_NAME = "claude-sonnet-4-6"
