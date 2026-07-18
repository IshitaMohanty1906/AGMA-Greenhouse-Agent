import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              classification_report)
import time
from step3_main_agent import run_agent, df

# ── Ground truth using your real column names ─────────────────
def get_ground_truth(row):
    temp     = float(row.get('indoor_temperature_c', 25))
    humidity = float(row.get('indoor_humidity_pct', 65))
    co2      = float(row.get('co2_ppm', 800))
    soil     = float(row.get('soil_moisture_pct', 50))
    light    = float(row.get('light_intensity_lux', 200))
    out_temp = float(row.get('outdoor_temperature_c', 20))

    # Parse timestamp hour for night/day rules
    try:
        ts   = str(row.get('timestamp', '2019-06-01 12:00:00'))
        hour = int(ts.split(' ')[1].split(':')[0])
    except:
        hour = 12

    # Ventilation: temp OR co2 OR humidity threshold
    ventilation = 1 if (temp > 28 or co2 > 1200 or humidity > 80) else 0

    # Irrigation: soil dry AND daytime AND not too humid
    irrigation  = 1 if (soil < 35 and 6 <= hour <= 20 and humidity < 85) else 0

    # Heating: too cold OR cold night
    heating     = 1 if (temp < 18 or (hour >= 22 and temp < 20)
                         or (hour <= 6  and temp < 18)) else 0

    # Lighting: low light AND daytime hours
    lighting    = 1 if (light < 150 and 6 <= hour <= 20) else 0

    return ventilation, irrigation, heating, lighting

# ── Parse agent decision from explanation text ────────────────
def parse_decision(result):
    text = result.get('explanation', '').lower()

    # Ventilation
    if 'ventilation on' in text or 'ventilat' in text and 'off' not in text.split('ventilat')[1][:20]:
        ventilation = 1
    else:
        ventilation = 0

    # Irrigation
    if 'irrigation on' in text or 'irrigat' in text and 'off' not in text.split('irrigat')[1][:20]:
        irrigation = 1
    else:
        irrigation = 0

    # Heating
    if 'heating on' in text or 'heat' in text and 'off' not in text.split('heat')[1][:20]:
        heating = 1
    else:
        heating = 0

    # Lighting
    if 'lighting on' in text or 'light' in text and 'off' not in text.split('light')[1][:20]:
        lighting = 1
    else:
        lighting = 0

    return ventilation, irrigation, heating, lighting

# ── Stratified sample — ~33 per month ────────────────────────
print("="*60)
print("AGMA EVALUATION — 200 STRATIFIED SAMPLES")
print("="*60)

np.random.seed(42)
total_rows  = len(df)
month_size  = total_rows // 6   # 6 months: April-September
sample_indices = []

for m in range(6):
    start = m * month_size
    end   = min(start + month_size, total_rows)
    picks = np.random.choice(range(start, end),
                             size=min(34, end - start),
                             replace=False)
    sample_indices.extend(picks.tolist())

sample_indices = sample_indices[:200]
print(f"Evaluating {len(sample_indices)} samples across 6 months...\n")

# ── Evaluation loop ───────────────────────────────────────────
y_true_v, y_true_i, y_true_h, y_true_l = [], [], [], []
y_pred_v, y_pred_i, y_pred_h, y_pred_l = [], [], [], []
response_times = []
errors = 0

for i, idx in enumerate(sample_indices):
    print(f"  Sample {i+1:3d}/200  (dataset row {idx})", end='\r')

    row = df.iloc[idx].to_dict()
    gt  = get_ground_truth(row)
    y_true_v.append(gt[0]); y_true_i.append(gt[1])
    y_true_h.append(gt[2]); y_true_l.append(gt[3])

    t0 = time.time()
    try:
        result = run_agent(idx)
        pred   = parse_decision(result)
    except Exception as e:
        pred = (0, 0, 0, 0)
        errors += 1
    elapsed = time.time() - t0
    response_times.append(elapsed)

    y_pred_v.append(pred[0]); y_pred_i.append(pred[1])
    y_pred_h.append(pred[2]); y_pred_l.append(pred[3])

# ── Aggregate metrics ─────────────────────────────────────────
all_true = y_true_v + y_true_i + y_true_h + y_true_l
all_pred = y_pred_v + y_pred_i + y_pred_h + y_pred_l

print("\n\n" + "="*60)
print("AGMA EVALUATION RESULTS")
print("="*60)
print(f"Samples evaluated : {len(sample_indices)}")
print(f"Errors            : {errors}")
print(f"Avg Response Time : {np.mean(response_times):.2f} seconds")
print(f"Min Response Time : {np.min(response_times):.2f} seconds")
print(f"Max Response Time : {np.max(response_times):.2f} seconds")
print()
print(f"Overall Accuracy  : {accuracy_score(all_true, all_pred)*100:.2f}%")
print(f"Macro F1-Score    : {f1_score(all_true, all_pred, average='macro')*100:.2f}%")
print(f"Macro Precision   : {precision_score(all_true, all_pred, average='macro', zero_division=0)*100:.2f}%")
print(f"Macro Recall      : {recall_score(all_true, all_pred, average='macro', zero_division=0)*100:.2f}%")
print()
print("Per-class Accuracy:")
for name, yt, yp in [
    ("Ventilation", y_true_v, y_pred_v),
    ("Irrigation",  y_true_i, y_pred_i),
    ("Heating",     y_true_h, y_pred_h),
    ("Lighting",    y_true_l, y_pred_l),
]:
    acc = accuracy_score(yt, yp) * 100
    f1  = f1_score(yt, yp, zero_division=0) * 100
    print(f"  {name:<12}: Accuracy={acc:.2f}%  F1={f1:.2f}%")

print()
print("Full Classification Report:")
labels = ['Ventilation','Irrigation','Heating','Lighting']
for name, yt, yp in [
    ("Ventilation", y_true_v, y_pred_v),
    ("Irrigation",  y_true_i, y_pred_i),
    ("Heating",     y_true_h, y_pred_h),
    ("Lighting",    y_true_l, y_pred_l),
]:
    print(f"\n{name}:")
    print(classification_report(yt, yp, zero_division=0))

print("="*60)
print("Evaluation complete. Take screenshot of results above.")
print("="*60)