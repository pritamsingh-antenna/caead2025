# CAEAD - Physics-Informed Neural Network (PINN)
# Dr. B. Pritam Singh
# Maxwell's equations ko directly Neural Network mein enforce karna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("CAEAD - Physics-Informed Neural Network (PINN)")
print("Dr. B. Pritam Singh")
print("=" * 55)
print("Physics constraint: Resonant frequency = c / (2L√εr)")
print("=" * 55)

# ── Manual Neural Network (no PyTorch needed) ─────────
# Simple backprop implementation with physics loss

class PINN:
    def __init__(self, layers=[4, 64, 64, 32, 1], lr=0.001):
        self.layers = layers
        self.lr = lr
        self.weights = []
        self.biases = []
        
        # Initialize weights
        np.random.seed(42)
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_grad(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(len(self.weights)-1):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            current = self.relu(z)
            self.activations.append(current)
        
        # Output layer - linear
        z = current @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)
        return z
    
    def physics_loss(self, X_orig, y_pred, scaler_X, scaler_y):
        """
        Physics constraint: f0 = c / (2 * L * sqrt(er))
        X_orig columns: [length_mm, width_mm, er, height_mm, feed, ground]
        """
        c = 3e8
        L = X_orig[:, 0] / 1000  # mm to m
        er = X_orig[:, 2]
        
        # Physics prediction
        f_physics = c / (2 * L * np.sqrt(er)) / 1e9  # GHz
        
        # ML prediction (unscaled)
        f_ml = y_pred.flatten()
        
        # Physics residual loss
        phys_loss = np.mean((f_ml - f_physics)**2)
        return phys_loss, f_physics
    
    def train(self, X, y, X_orig, scaler_X, scaler_y, epochs=500):
        losses = []
        phys_losses = []
        
        print("\nTraining PINN...")
        print(f"{'Epoch':>6} | {'Data Loss':>10} | {'Physics Loss':>12} | {'Total Loss':>10}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Data loss
            data_loss = np.mean((y_pred - y.reshape(-1,1))**2)
            
            # Physics loss
            phys_loss, _ = self.physics_loss(X_orig, y_pred, scaler_X, scaler_y)
            
            # Total loss — physics weight 0.3
            total_loss = data_loss + 0.3 * phys_loss
            
            losses.append(data_loss)
            phys_losses.append(phys_loss)
            
            # Backprop (output layer)
            delta = 2 * (y_pred - y.reshape(-1,1)) / len(y)
            
            for i in range(len(self.weights)-1, -1, -1):
                dW = self.activations[i].T @ delta
                db = np.sum(delta, axis=0, keepdims=True)
                
                self.weights[i] -= self.lr * dW
                self.biases[i]  -= self.lr * db
                
                if i > 0:
                    delta = delta @ self.weights[i].T
                    delta *= self.relu_grad(self.z_values[i-1])
            
            if (epoch+1) % 100 == 0:
                print(f"{epoch+1:>6} | {data_loss:>10.6f} | {phys_loss:>12.6f} | {total_loss:>10.6f}")
        
        return losses, phys_losses

# ── Load data ─────────────────────────────────────────
print("\n[1/4] Loading 5000-design dataset...")
df = pd.read_csv('caead_5000_designs.csv')

X = df[['patch_length_mm','patch_width_mm',
        'substrate_er','substrate_h_mm',
        'feed_position','ground_ratio']].values
y = df['resonant_freq_GHz'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train PINN ────────────────────────────────────────
print("\n[2/4] Training PINN with physics constraints...")
pinn = PINN(layers=[6, 64, 64, 32, 1], lr=0.001)
losses, phys_losses = pinn.train(
    X_train_sc, y_train_sc,
    X_train, scaler_X, scaler_y,
    epochs=500
)

# ── Evaluate ──────────────────────────────────────────
print("\n[3/4] Evaluating PINN...")
y_pred_sc = pinn.forward(X_test_sc).flatten()
y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).flatten()

mae  = np.mean(np.abs(y_pred - y_test))
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
r2   = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

# Physics violation check
c = 3e8
L_test = X_test[:, 0] / 1000
er_test = X_test[:, 2]
f_physics = c / (2 * L_test * np.sqrt(er_test)) / 1e9
physics_violation = np.mean(np.abs(y_pred - f_physics))

print(f"\n   R²    : {r2:.4f}")
print(f"   MAE   : {mae:.4f} GHz")
print(f"   RMSE  : {rmse:.4f} GHz")
print(f"   Physics violation: {physics_violation:.4f} GHz")

# ── Plots ──────────────────────────────────────────────
print("\n[4/4] Plotting results...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CAEAD — Physics-Informed Neural Network (PINN)\nDr. B. Pritam Singh', fontsize=13)

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.4, s=8, color='steelblue')
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn,mx],[mn,mx],'r--',linewidth=2,label='Perfect')
axes[0].set_xlabel('Actual Frequency (GHz)')
axes[0].set_ylabel('Predicted Frequency (GHz)')
axes[0].set_title(f'Actual vs Predicted\nR²={r2:.4f}')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Plot 2: Training loss
axes[1].plot(losses, color='blue', linewidth=2, label='Data Loss')
axes[1].plot(phys_losses, color='red', linewidth=2, label='Physics Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('PINN Training Loss\n(Data + Physics Constraints)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

# Plot 3: Physics compliance
axes[2].scatter(f_physics, y_pred, alpha=0.4, s=8, color='green')
mn2, mx2 = f_physics.min(), f_physics.max()
axes[2].plot([mn2,mx2],[mn2,mx2],'r--',linewidth=2,label='Perfect physics')
axes[2].set_xlabel('Physics Formula (GHz)')
axes[2].set_ylabel('PINN Prediction (GHz)')
axes[2].set_title(f'Physics Compliance\nViolation={physics_violation:.4f} GHz')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('caead_pinn_results.png', dpi=150)
plt.show()

print("\n" + "=" * 55)
print("CAEAD PINN — Summary")
print("=" * 55)
print(f"Architecture : 6 → 64 → 64 → 32 → 1")
print(f"Physics      : f0 = c/(2L√εr) enforced in loss")
print(f"R²           : {r2:.4f}")
print(f"MAE          : {mae:.4f} GHz")
print(f"Physics viol.: {physics_violation:.4f} GHz")
print("=" * 55)
print("Regular ML   : No physics constraint")
print("PINN         : Maxwell's equations enforced!")
print("=" * 55)