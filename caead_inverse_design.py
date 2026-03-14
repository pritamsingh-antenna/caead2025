# CAEAD - Inverse Design
# Dr. B. Pritam Singh
# Target specs do — AI geometry batayega

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("CAEAD - Inverse Design System")
print("Dr. B. Pritam Singh")
print("=" * 50)

# ── Load dataset ──────────────────────────────────
print("\n[1/4] Loading dataset...")
df = pd.read_csv('caead_5000_designs.csv')

# ── INVERSE: specs → geometry ─────────────────────
# Input  = what you WANT  (freq, bandwidth, gain)
# Output = what to BUILD  (length, width, er, height)

X_inv = df[['resonant_freq_GHz', 'bandwidth_MHz', 'gain_dBi']].values
y_inv = df[['patch_length_mm', 'patch_width_mm', 
            'substrate_er', 'substrate_h_mm']].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X_inv, y_inv, test_size=0.2, random_state=42
)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_tr_sc = scaler_x.fit_transform(X_tr)
X_te_sc = scaler_x.transform(X_te)
y_tr_sc = scaler_y.fit_transform(y_tr)

# ── Train inverse model ───────────────────────────
print("[2/4] Training inverse design model...")

model = MLPRegressor(
    hidden_layer_sizes=(128, 128, 64, 32),
    activation='relu',
    max_iter=2000,
    random_state=42,
    learning_rate_init=0.001
)

model.fit(X_tr_sc, y_tr_sc)
print("   Done!")

# ── Test inverse model ────────────────────────────
print("\n[3/4] Testing inverse predictions...")

y_pred_sc = model.predict(X_te_sc)
y_pred    = scaler_y.inverse_transform(y_pred_sc)

# ── Demo: 3 target specs ──────────────────────────
print("\n[4/4] LIVE DEMO — 3 target designs:")
print("=" * 50)

targets = [
    {"freq": 2.4, "bw": 5.0,  "gain": 2.0, "name": "WiFi 2.4 GHz"},
    {"freq": 5.0, "bw": 10.0, "gain": 4.0, "name": "WiFi 5 GHz"},
    {"freq": 3.5, "bw": 20.0, "gain": 3.5, "name": "5G Sub-6 GHz"},
]

results = []
for t in targets:
    inp = np.array([[t['freq'], t['bw'], t['gain']]])
    inp_sc = scaler_x.transform(inp)
    out_sc = model.predict(inp_sc)
    out    = scaler_y.inverse_transform(out_sc)[0]
    
    # Clip to physical limits
    out[0] = np.clip(out[0], 10, 40)   # length
    out[1] = np.clip(out[1], 10, 40)   # width
    out[2] = np.clip(out[2], 2.2, 10.2) # er
    out[3] = np.clip(out[3], 0.8, 3.2)  # height
    
    results.append({
        'Application'    : t['name'],
        'Target Freq'    : f"{t['freq']} GHz",
        'Target BW'      : f"{t['bw']} MHz",
        'Target Gain'    : f"{t['gain']} dBi",
        'Length (mm)'    : f"{out[0]:.2f}",
        'Width (mm)'     : f"{out[1]:.2f}",
        'Substrate Er'   : f"{out[2]:.2f}",
        'Height (mm)'    : f"{out[3]:.2f}",
    })
    
    print(f"\n  {t['name']}")
    print(f"  Target  : {t['freq']} GHz | {t['bw']} MHz BW | {t['gain']} dBi")
    print(f"  Design  : L={out[0]:.2f}mm | W={out[1]:.2f}mm | Er={out[2]:.2f} | H={out[3]:.2f}mm")

# ── Plot comparison ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CAEAD — Inverse Design Results\nDr. B. Pritam Singh', fontsize=13)

labels = ['Patch Length (mm)', 'Patch Width (mm)', 
          'Substrate Er', 'Substrate H (mm)']
colors = ['steelblue', 'green', 'orange', 'purple']

for i, (label, color) in enumerate(zip(labels[:3], colors[:3])):
    axes[i].scatter(y_te[:, i], y_pred[:, i], 
                    alpha=0.3, s=8, color=color)
    mn, mx = y_te[:, i].min(), y_te[:, i].max()
    axes[i].plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    axes[i].set_xlabel(f'Actual {label}')
    axes[i].set_ylabel(f'Predicted {label}')
    axes[i].set_title(label)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('caead_inverse_design.png', dpi=150)
plt.show()

# Save results
pd.DataFrame(results).to_csv('caead_inverse_results.csv', index=False)

print("\n" + "=" * 50)
print("CAEAD Inverse Design — Complete!")
print("=" * 50)
print("Input  : Target Specs (Freq, BW, Gain)")
print("Output : Antenna Geometry (L, W, Er, H)")
print("Files  : caead_inverse_design.png")
print("         caead_inverse_results.csv")