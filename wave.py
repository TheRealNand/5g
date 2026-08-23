import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SAMPLES = 5000
DT = 0.001                 # 1 ms between observations
SEED = 42

np.random.seed(SEED)

OUTPUT_DIR = Path("data")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# TIME
# ============================================================

time = np.arange(NUM_SAMPLES) * DT


# ============================================================
# 1. SLOW CHANNEL VARIATION
# ============================================================

# Represents gradual changes caused by movement/environment.

slow_variation = (
    3.0 * np.sin(2 * np.pi * 0.15 * time)
    + 1.5 * np.sin(2 * np.pi * 0.45 * time)
)


# ============================================================
# 2. TIME-CORRELATED FADING
# ============================================================

# Generate random fading and smooth it so that adjacent
# channel observations are correlated.

random_component = np.random.normal(0, 1, NUM_SAMPLES)

fading = np.zeros(NUM_SAMPLES)

correlation = 0.96

for i in range(1, NUM_SAMPLES):
    fading[i] = (
        correlation * fading[i - 1]
        + np.sqrt(1 - correlation**2) * random_component[i]
    )

fading = fading / np.std(fading)

fading_db = 3.0 * fading


# ============================================================
# 3. CONTROLLED CHANNEL DETERIORATION
# ============================================================

# We deliberately create a deterioration region.
# This allows us to test whether prediction can anticipate it.

deterioration = np.zeros(NUM_SAMPLES)

start = int(NUM_SAMPLES * 0.55)
end = int(NUM_SAMPLES * 0.70)

deterioration[start:end] = np.linspace(
    0,
    -10,
    end - start
)

# Recover afterwards

deterioration[end:] = -10 * np.exp(
    -(np.arange(NUM_SAMPLES - end) / 250)
)


# ============================================================
# 4. RECEIVED SIGNAL POWER / RSRP
# ============================================================

baseline_rsrp = -75

rsrp = (
    baseline_rsrp
    + slow_variation
    + fading_db
    + deterioration
)


# ============================================================
# 5. INTERFERENCE
# ============================================================

interference = (
    -100
    + 2 * np.sin(2 * np.pi * 0.08 * time)
)

# Add an interference burst

burst_start = int(NUM_SAMPLES * 0.40)
burst_end = int(NUM_SAMPLES * 0.48)

interference[burst_start:burst_end] += 8


# ============================================================
# 6. NOISE
# ============================================================

noise_power = -100


# ============================================================
# 7. CALCULATE SINR
# ============================================================

def db_to_linear(db):
    return 10 ** (db / 10)


def linear_to_db(linear):
    return 10 * np.log10(linear)


signal_linear = db_to_linear(rsrp)

interference_linear = db_to_linear(interference)

noise_linear = db_to_linear(noise_power)

sinr_linear = signal_linear / (
    interference_linear + noise_linear
)

sinr_db = linear_to_db(sinr_linear)


# ============================================================
# 8. APPROXIMATE CQI MAPPING
# ============================================================

# IMPORTANT:
# This is a prototype mapping, NOT a 3GPP-compliant CQI table.
# We will replace it with the appropriate standard mapping
# once the channel/link model is finalized.

cqi_thresholds = np.array([
    -5, -3, -1, 1,
     3,  5,  7, 9,
    11, 13, 15, 17,
    19, 21, 23
])

cqi = np.digitize(sinr_db, cqi_thresholds)

cqi = np.clip(cqi, 0, 15)


# ============================================================
# 9. CREATE DATASET
# ============================================================

df = pd.DataFrame({
    "time_s": time,
    "rsrp_dbm": rsrp,
    "sinr_db": sinr_db,
    "cqi": cqi
})

csv_path = OUTPUT_DIR / "channel_dataset.csv"

df.to_csv(csv_path, index=False)

print(f"Dataset saved to: {csv_path}")
print()
print(df.head())


# ============================================================
# 10. VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 9),
    sharex=True
)


# ------------------------------------------------------------
# RSRP
# ------------------------------------------------------------

axes[0].plot(
    time,
    rsrp,
    color="#C32626",
    linewidth=1.8
)

axes[0].set_ylabel("RSRP (dBm)")
axes[0].set_title("Time-Varying Wireless Channel")
axes[0].grid(True)


# ------------------------------------------------------------
# SINR
# ------------------------------------------------------------

axes[1].plot(
    time,
    sinr_db,
    color="#510BF5",
    linewidth=1.8
)

axes[1].set_ylabel("SINR (dB)")
axes[1].grid(True)


# ------------------------------------------------------------
# CQI
# ------------------------------------------------------------

axes[2].step(
    time,
    cqi,
    where="post",
    color="#16A34A",
    linewidth=1.8
)

axes[2].set_ylabel("CQI")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)


# ------------------------------------------------------------
# FINAL LAYOUT
# ------------------------------------------------------------

plt.tight_layout()

plot_path = RESULTS_DIR / "channel_quality_vs_time.png"

plt.savefig(
    plot_path,
    dpi=200
)

plt.show()

print(f"\nPlot saved to: {plot_path}")