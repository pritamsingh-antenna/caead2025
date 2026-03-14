# CAEAD - ML Surrogate Model
# Dr. B. Pritam Singh
# 500 antenna designs + DNN training

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("CAEAD - Surrogate Model Training")
print("Dr. B. Pritam Singh")
print("=" * 50)

# ── STEP 1: Generate 500 designs ──────────────────
print("\n[1/4] Generating 500 antenna designs...")

c = 3e8
np.random.seed(42)

# Random patch parameters
patch_lengths = np.random.uniform(10e-3, 40e-3, 500)  # 10-40mm
patch_widths  = np.random.uniform(10e-3, 40e-3, 500)  # 10-40mm
substrates_er = np.random.uniform(2.2, 10.2, 500)      # er range
substrates_h  = np.random.uniform(0.8e-3, 3.2e-3, 500) # height 0.8-3.2mm

# Calculate resonant frequency
f0 = c / (2 * patch_lengths * np.sqrt(substrates_er))

# Bandwidth ~ proportional to h/L
BW = 0.023 * f0 * (substrates_h / patch_lengths)

# Min S11 (simplified)
min_s11 = np.random.uniform(-35, -10, 500)

# Build dataset
df = pd.DataFrame({
    'patch_length_mm' : patch_lengths * 1000,
    'patch_width_mm'  : patch_widths  * 1000,
    'substrate_er'    : substrates_er,
    'substrate_h_mm'  : substrates_h  * 1000,
    'resonant_freq_GHz': f0 / 1e9,
    'bandwidth_MHz'   : BW / 1e6,
    'min_s11_dB'      : min_s11
})

df.to_csv('caead_500_designs.csv', index=False)
print(f"   Done! {len(df)} designs generated.")

# ── STEP 2: Prepare ML data ───────────────────────
print("\n[2/4] Preparing data for ML training...")

X = df[['patch_length_mm','patch_width_mm','substrate_er','substrate_h_mm']].values
y = df['resonant_freq_GHz'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── STEP 3: Train DNN ─────────────────────────────
print("\n[3/4] Training DNN surrogate model...")

model = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),
    activation='relu',
    max_iter=1000,
    random_state=42,
    verbose=False
)

model.fit(X_train_sc, y_train)
print("   Training complete!")

# ── STEP 4: Results ───────────────────────────────
print("\n[4/4] Evaluating model...")

y_pred = model.predict(X_test_sc)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n   MAE  : {mae:.4f} GHz")
print(f"   R²   : {r2:.4f}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='navy', s=50)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual Frequency (GHz)')
plt.ylabel('Predicted Frequency (GHz)')
plt.title('CAEAD - DNN Surrogate Model\nActual vs Predicted Resonant Frequency\nDr. B. Pritam Singh')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('caead_ml_results.png')
plt.show()

print("\n" + "=" * 50)
print("CAEAD Surrogate Model - Summary")
print("=" * 50)
print(f"Dataset    : 500 antenna designs")
print(f"Features   : Length, Width, Er, Height")
print(f"Target     : Resonant Frequency")
print(f"Model      : DNN (64-64-32)")
print(f"MAE        : {mae:.4f} GHz")
print(f"R² Score   : {r2:.4f}")
print(f"Accuracy   : {(1-mae/np.mean(y_test))*100:.1f}%")
print("=" * 50)
print("Files: caead_500_designs.csv, caead_ml_results.png")