# CAEAD - Large Dataset Generation
# Dr. B. Pritam Singh
# 500 → 5000 antenna designs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("CAEAD - Large Dataset Generation")
print("Dr. B. Pritam Singh")
print("=" * 50)

c = 3e8
np.random.seed(123)

N = 5000  # designs

print(f"\nGenerating {N} antenna designs...")

# 6 parameters — Latin Hypercube style uniform sampling
patch_lengths  = np.random.uniform(10e-3, 40e-3, N)
patch_widths   = np.random.uniform(10e-3, 40e-3, N)
substrates_er  = np.random.uniform(2.2, 10.2, N)
substrates_h   = np.random.uniform(0.8e-3, 3.2e-3, N)
feed_positions = np.random.uniform(0.1, 0.4, N)   # feed offset ratio
ground_sizes   = np.random.uniform(1.2, 2.0, N)   # ground plane ratio

# Physics-based calculations
f0  = c / (2 * patch_lengths * np.sqrt(substrates_er))
BW  = 0.023 * f0 * (substrates_h / patch_lengths) * (1 + 0.1 * ground_sizes)
gain = 6.5 + 2 * np.log10(patch_widths / patch_lengths) - 0.5 * substrates_er + np.random.normal(0, 0.3, N)
efficiency = 85 - 5 * substrates_er + 10 * (substrates_h / 3.2e-3) + np.random.normal(0, 2, N)
efficiency = np.clip(efficiency, 40, 98)
min_s11 = np.random.uniform(-40, -10, N)

# Build dataframe
df = pd.DataFrame({
    'design_id'         : range(1, N+1),
    'patch_length_mm'   : patch_lengths * 1000,
    'patch_width_mm'    : patch_widths  * 1000,
    'substrate_er'      : substrates_er,
    'substrate_h_mm'    : substrates_h  * 1000,
    'feed_position'     : feed_positions,
    'ground_ratio'      : ground_sizes,
    'resonant_freq_GHz' : f0 / 1e9,
    'bandwidth_MHz'     : BW / 1e6,
    'gain_dBi'          : gain,
    'efficiency_pct'    : efficiency,
    'min_s11_dB'        : min_s11
})

df.to_csv('caead_5000_designs.csv', index=False)
print(f"Done! {N} designs saved.")

# ── Stats ──────────────────────────────────────────
print("\n" + "=" * 50)
print("Dataset Statistics:")
print("=" * 50)
print(f"Frequency range : {df['resonant_freq_GHz'].min():.2f} — {df['resonant_freq_GHz'].max():.2f} GHz")
print(f"Bandwidth range : {df['bandwidth_MHz'].min():.1f} — {df['bandwidth_MHz'].max():.1f} MHz")
print(f"Gain range      : {df['gain_dBi'].min():.1f} — {df['gain_dBi'].max():.1f} dBi")
print(f"Efficiency range: {df['efficiency_pct'].min():.1f} — {df['efficiency_pct'].max():.1f} %")
print(f"Parameters      : 6 (L, W, Er, H, Feed, Ground)")
print(f"Total designs   : {N}")

# ── Plots ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('CAEAD — 5000 Antenna Dataset\nDr. B. Pritam Singh', fontsize=14)

axes[0,0].hist(df['resonant_freq_GHz'], bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
axes[0,0].set_xlabel('Resonant Frequency (GHz)')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Frequency Distribution')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].hist(df['bandwidth_MHz'], bins=50, color='green', edgecolor='darkgreen', alpha=0.7)
axes[0,1].set_xlabel('Bandwidth (MHz)')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Bandwidth Distribution')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].scatter(df['resonant_freq_GHz'], df['gain_dBi'],
                  alpha=0.3, s=5, color='orange')
axes[1,0].set_xlabel('Resonant Frequency (GHz)')
axes[1,0].set_ylabel('Gain (dBi)')
axes[1,0].set_title('Frequency vs Gain')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].scatter(df['substrate_er'], df['efficiency_pct'],
                  alpha=0.3, s=5, color='purple')
axes[1,1].set_xlabel('Substrate Er')
axes[1,1].set_ylabel('Efficiency (%)')
axes[1,1].set_title('Substrate Er vs Efficiency')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('caead_5000_dataset_plots.png', dpi=150)
plt.show()

print("\nFiles saved:")
print("  caead_5000_designs.csv")
print("  caead_5000_dataset_plots.png")
print("=" * 50)