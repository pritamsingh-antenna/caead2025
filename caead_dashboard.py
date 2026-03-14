# CAEAD - Interactive Dashboard
# Dr. B. Pritam Singh
# Target frequency dो — AI best antenna design dhundega LIVE!

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="CAEAD - AI Antenna Designer",
    page_icon="📡",
    layout="wide"
)

# ── Header ────────────────────────────────────────
st.title("📡 CAEAD — AI-Driven Antenna Design")
st.subheader("Dr. B. Pritam Singh | IIT (ISM) Dhanbad")
st.markdown("*Center for AI-Driven Electromagnetics & Antenna Design*")
st.divider()

# ── Load & Train (cached) ─────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv('caead_500_designs.csv')
    X  = df[['patch_length_mm','patch_width_mm',
              'substrate_er','substrate_h_mm']].values
    y  = df['resonant_freq_GHz'].values
    
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation='relu',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_sc, y)
    return model, scaler, df

model, scaler, df = load_and_train()

# ── Sidebar inputs ────────────────────────────────
st.sidebar.header("🎯 Design Target")
target_freq = st.sidebar.slider(
    "Target Frequency (GHz)",
    min_value=1.0, max_value=10.0,
    value=5.0, step=0.1
)

iterations = st.sidebar.slider(
    "Optimization Iterations",
    min_value=100, max_value=1000,
    value=200, step=100
)

run = st.sidebar.button("🚀 Run CAEAD Optimizer", type="primary")

# ── Main content ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10), width='stretch)
    st.caption(f"Total designs in dataset: {len(df)}")

with col2:
    st.subheader("📈 Frequency Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(df['resonant_freq_GHz'], bins=30, 
             color='steelblue', edgecolor='navy', alpha=0.7)
    ax1.axvline(x=target_freq, color='red', 
                linestyle='--', linewidth=2,
                label=f'Target: {target_freq} GHz')
    ax1.set_xlabel('Resonant Frequency (GHz)')
    ax1.set_ylabel('Count')
    ax1.set_title('Dataset Frequency Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

st.divider()

# ── Optimizer ─────────────────────────────────────
if run:
    st.subheader(f"🔍 Optimizing for {target_freq} GHz...")
    
    progress = st.progress(0)
    
    best_design = None
    best_error  = 999
    best_freq   = 0
    history     = []
    
    for i in range(iterations):
        candidate = np.array([[
            np.random.uniform(10, 40),
            np.random.uniform(10, 40),
            np.random.uniform(2.2, 10.2),
            np.random.uniform(0.8, 3.2)
        ]])
        
        candidate_sc = scaler.transform(candidate)
        predicted_f  = model.predict(candidate_sc)[0]
        error        = abs(predicted_f - target_freq)
        
        history.append(error)
        
        if error < best_error:
            best_error  = error
            best_design = candidate[0]
            best_freq   = predicted_f
        
        progress.progress((i+1)/iterations)
    
    # ── Results ───────────────────────────────────
    st.success(f"✅ Optimization Complete!")
    
    col3, col4, col5 = st.columns(3)
    col3.metric("Target Frequency", f"{target_freq} GHz")
    col4.metric("Predicted Frequency", f"{best_freq:.3f} GHz")
    col5.metric("Error", f"{best_error/target_freq*100:.2f}%")
    
    st.subheader("🏆 Best Antenna Design Found:")
    
    res = pd.DataFrame({
        'Parameter'  : ['Patch Length','Patch Width',
                        'Substrate Er','Substrate Height'],
        'Value'      : [f"{best_design[0]:.2f} mm",
                        f"{best_design[1]:.2f} mm",
                        f"{best_design[2]:.2f}",
                        f"{best_design[3]:.2f} mm"]
    })
    st.dataframe(res, width='stretch, hide_index=True)
    
    # Convergence plot
    st.subheader("📉 Optimization Convergence")
    best_so_far = pd.Series(history).cummin()
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(best_so_far, color='blue', linewidth=2)
    ax2.axhline(y=0.05, color='red', linestyle='--', 
                linewidth=2, label='Target: <0.05 GHz')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Error (GHz)')
    ax2.set_title(f'Convergence — Target: {target_freq} GHz')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    st.info("⚡ Traditional HFSS: 2-8 hours | CAEAD AI: < 1 second")

else:
    st.info("👈 Set target frequency in sidebar and click **Run CAEAD Optimizer**")

# ── Footer ────────────────────────────────────────
st.divider()
st.caption("CAEAD — Center for AI-Driven Electromagnetics & Antenna Design | Dr. B. Pritam Singh | IIT (ISM) Dhanbad")