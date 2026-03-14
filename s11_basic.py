# CAEAD - S11 Calculation Basics
# Dr. B. Pritam Singh

import numpy as np
import matplotlib.pyplot as plt

# Frequency range - 1 GHz to 6 GHz
freq = np.linspace(1e9, 6e9, 500)

# Patch antenna resonant frequency - 2.4 GHz (WiFi band)
f0 = 2.4e9

# Bandwidth
BW = 0.1e9  # 100 MHz

# Simple S11 model (Lorentzian curve)
s11_linear = (freq - f0)**2 / ((freq - f0)**2 + (BW/2)**2)
s11_dB = 20 * np.log10(s11_linear + 1e-10)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(freq/1e9, s11_dB, color='blue', linewidth=2)
plt.axhline(y=-10, color='red', linestyle='--', label='-10 dB threshold')
plt.axvline(x=2.4, color='green', linestyle='--', label='2.4 GHz resonance')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.title('CAEAD - Patch Antenna S11 Response\nDr. B. Pritam Singh')
plt.legend()
plt.grid(True)
plt.ylim(-40, 5)
plt.tight_layout()
plt.savefig('s11_plot.png')
plt.show()

print("S11 plot saved!")
print(f"Resonant frequency: {f0/1e9} GHz")
print(f"Min S11: {min(s11_dB):.2f} dB")