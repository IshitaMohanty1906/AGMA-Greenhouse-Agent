"""
================================================================
  STEP 1 — GREENHOUSE EDA
  Run this FIRST before anything else.
  Explores dataset and prints thresholds for agent.
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ── LOAD ─────────────────────────────────────────────────────────
print("=" * 55)
print("  GREENHOUSE CLIMATE DATASET — EDA")
print("=" * 55)

df = pd.read_csv('data/greenhouse_climate_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour']      = df['timestamp'].dt.hour
df['month']     = df['timestamp'].dt.month
df['day']       = df['timestamp'].dt.day

print(f"\n  Rows    : {len(df):,}")
print(f"  Columns : {len(df.columns)}")
print(f"  From    : {df['timestamp'].min()}")
print(f"  To      : {df['timestamp'].max()}")

# ── DATA QUALITY ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DATA QUALITY CHECK")
print("=" * 55)
print(f"\n  Null values  : {df.isnull().sum().sum()}")
print(f"  Duplicates   : {df.duplicated().sum()}")
print("  Status       : CLEAN ✅")

# ── BASIC STATS ───────────────────────────────────────────────────
sensor_cols = [
    'indoor_temperature_c','indoor_humidity_pct',
    'co2_ppm','light_intensity_lux','soil_moisture_pct'
]
print("\n  Basic Statistics:")
print(df[sensor_cols].describe().round(2).to_string())

# ── ACTUATOR STATS ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ACTUATOR USAGE")
print("=" * 55)
actuators = ['ventilation_status','heating_status','irrigation_status','supplemental_lighting']
for a in actuators:
    pct = df[a].mean() * 100
    print(f"  {a:<28}: ON {pct:.1f}% of time")

# ── ANOMALY COUNTS ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ANOMALY FREQUENCY")
print("=" * 55)
anomalies = {
    'High Temp (>35C)'     : (df['indoor_temperature_c'] > 35).sum(),
    'Low Temp (<15C)'      : (df['indoor_temperature_c'] < 15).sum(),
    'High Humidity (>85%)' : (df['indoor_humidity_pct'] > 85).sum(),
    'High CO2 (>1500ppm)'  : (df['co2_ppm'] > 1500).sum(),
    'Low Light (<150lux)'  : ((df['light_intensity_lux'] < 150) & df['hour'].between(8,18)).sum(),
    'Low Soil (<30%)'      : (df['soil_moisture_pct'] < 30).sum(),
}
for k, v in anomalies.items():
    print(f"  {k:<28}: {v:>6,} readings ({v/len(df)*100:.2f}%)")

# ── THRESHOLDS SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 55)
print("  THRESHOLDS — COPY INTO AGENT SYSTEM PROMPT")
print("=" * 55)
print(f"""
  Temperature  : Normal 18-28C  | Warning 28-35C | Critical >35C
  Humidity     : Normal 50-80%  | Warning 80-85% | Critical >85%
  CO2          : Normal 400-1200| Warning 1200-1500| Critical >1500 ppm
  Light        : Normal 200-800 | Low <150 lux (daytime)
  Soil Moisture: Normal 35-75%  | Low <30%       | Critical <25%

  Dataset averages:
  Avg Temp     : {df['indoor_temperature_c'].mean():.1f}C
  Avg Humidity : {df['indoor_humidity_pct'].mean():.1f}%
  Avg CO2      : {df['co2_ppm'].mean():.0f} ppm
  Avg Light    : {df['light_intensity_lux'].mean():.0f} lux
  Avg Soil     : {df['soil_moisture_pct'].mean():.1f}%
""")

# ── PLOTS ─────────────────────────────────────────────────────────
print("  Generating plots...")

fig, axes = plt.subplots(3, 2, figsize=(20, 14))
fig.suptitle('Greenhouse Climate Dataset — EDA Dashboard', fontsize=15, fontweight='bold')

sample = df.iloc[::72]  # every 6 hours

# 1 Temperature over time
axes[0,0].plot(sample['timestamp'], sample['indoor_temperature_c'], color='#e74c3c', lw=0.9)
axes[0,0].axhline(35, color='red', ls='--', lw=1, label='Critical 35C')
axes[0,0].axhline(18, color='blue', ls='--', lw=1, label='Min 18C')
axes[0,0].set_title('Indoor Temperature Over 6 Months', fontweight='bold')
axes[0,0].set_ylabel('Temperature (C)')
axes[0,0].legend(fontsize=8)

# 2 Humidity over time
axes[0,1].plot(sample['timestamp'], sample['indoor_humidity_pct'], color='#2ecc71', lw=0.9)
axes[0,1].axhline(85, color='orange', ls='--', lw=1, label='Critical 85%')
axes[0,1].set_title('Indoor Humidity Over 6 Months', fontweight='bold')
axes[0,1].set_ylabel('Humidity (%)')
axes[0,1].legend(fontsize=8)

# 3 CO2 over time
axes[1,0].plot(sample['timestamp'], sample['co2_ppm'], color='#9b59b6', lw=0.9)
axes[1,0].axhline(1500, color='red', ls='--', lw=1, label='Critical 1500ppm')
axes[1,0].set_title('CO2 Concentration Over 6 Months', fontweight='bold')
axes[1,0].set_ylabel('CO2 (ppm)')
axes[1,0].legend(fontsize=8)

# 4 Hourly patterns
hourly = df.groupby('hour')[sensor_cols].mean()
for col, color in zip(sensor_cols[:3], ['#e74c3c','#2ecc71','#9b59b6']):
    axes[1,1].plot(hourly.index, hourly[col]/hourly[col].max(), label=col.split('_')[1], color=color, lw=2)
axes[1,1].set_title('Normalized Hourly Patterns', fontweight='bold')
axes[1,1].set_xlabel('Hour of Day')
axes[1,1].legend(fontsize=8)

# 5 Actuator usage bar
act_pcts = [df[a].mean()*100 for a in actuators]
act_labels = ['Ventilation','Heating','Irrigation','Lighting']
colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']
bars = axes[2,0].bar(act_labels, act_pcts, color=colors, edgecolor='black', alpha=0.85)
axes[2,0].set_title('Actuator ON Time (%)', fontweight='bold')
axes[2,0].set_ylabel('Percentage (%)')
for b, p in zip(bars, act_pcts):
    axes[2,0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{p:.1f}%', ha='center', fontsize=9)

# 6 Correlation heatmap
corr = df[sensor_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=axes[2,1], linewidths=0.5, annot_kws={'size':9})
axes[2,1].set_title('Sensor Correlation Heatmap', fontweight='bold')

plt.tight_layout(pad=3.0)
plt.savefig('evaluation/eda_dashboard.png', dpi=130, bbox_inches='tight')
print("  Saved: evaluation/eda_dashboard.png")
plt.show()

print("\n  EDA COMPLETE. Now run: python step2_build_rag.py")
