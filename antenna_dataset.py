# CAEAD - Antenna Dataset Generation
# Dr. B. Pritam Singh
# Multiple patch antenna designs - S11 vs Frequency

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Patch antenna parameters
# Length (L) controls resonant frequency: f0 = c / (2 * L * sqrt(er))
# er = dielectric constant of substrate

c = 3e8  # speed of light
er = 4.4  # FR4 substrate

# Generate 20 different patch lengths - 10mm to 40mm
patch_lengths = np.linspace(10e-3, 40e-3, 20)

# Frequency range
freq = np.linspace(1e9, 10e9, 500)

# Store results
results = []

plt.figure(figsize=(12, 6))

for i, L in enumerate(patch_lengths):
    # Resonant frequency formula
    f0 = c / (2 * L * np.sqrt(er))
    
    # Bandwidth ~ 2% of f0 for basic patch
    BW = 0.02 * f0
    
    # S11 model
    s11_linear = (freq - f0)**2 / ((freq - f0)**2 + (BW/2)**2)
    s11_dB = 20 * np.log10(s11_linear + 1e-10)
    
    # Save to results
    results.append({
        'design_id': i+1,
        'patch_length_mm': round(L*1000, 2),
        'resonant_freq_GHz': round(f0/1e9, 3),
        'min_s11_dB': round(min(s11_dB), 2),
        'bandwidth_MHz': round(BW/1e6, 1)
    })
    
    # Plot
    plt.plot(freq/1e9, s11_dB, alpha=0.6, linewidth=1.5)

plt.axhline(y=-10, color='red', linestyle='--', linewidth=2, label='-10 dB threshold')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.title('CAEAD - Antenna Dataset: 20 Patch Designs\nDr. B. Pritam Singh')
plt.legend()
plt.grid(True)
plt.ylim(-40, 5)
plt.tight_layout()
plt.savefig('antenna_dataset_plot.png')
plt.show()

# Save dataset to CSV
df = pd.DataFrame(results)
df.to_csv('antenna_dataset.csv', index=False)

print("=" * 50)
print("CAEAD Dataset Generated!")
print("=" * 50)
print(df.to_string(index=False))
print("=" * 50)
print(f"Total designs: {len(results)}")
print("Files saved: antenna_dataset_plot.png, antenna_dataset.csv")