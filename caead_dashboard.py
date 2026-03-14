# CAEAD - Interactive Dashboard v2
# Dr. B. Pritam Singh
# Forward Design + Inverse Design — dono ek jagah!

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="CAEAD - AI Antenna Designer",
    page_icon="📡",
    layout="wide"
)

st.title("📡 CAEAD — AI-Driven Antenna Design")
st.subheader("Dr. B. Pritam Singh | IIT (ISM) Dhanbad")
st.markdown("*Center for AI-Driven Electromagnetics & Antenna Design*")
st.divider()

# ── Load & Train ──────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv('caead_5000_designs.csv')
    
    X = df[['patch_length_mm','patch_width_mm',
            'substrate_er','substrate_h_mm',
            'feed_position','ground_ratio']].values
    y_freq = df['resonant_freq_GHz'].values
    y_bw   = df['bandwidth_MHz'].values
    y_gain = df['gain_dBi'].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    m_freq = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=1000, random_state=42)
    m_bw   = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=1000, random_state=42)
    m_gain = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=1000, random_state=42)

    m_freq.fit(X_sc, y_freq)
    m_bw.fit(X_sc, y_bw)
    m_gain.fit(X_sc, y_gain)

    # Inverse model
    X_inv = df[['resonant_freq_GHz','bandwidth_MHz','gain_dBi']].values
    y_inv = df[['patch_length_mm','patch_width_mm','substrate_er','substrate_h_mm']].values

    sc_x = StandardScaler(); sc_y = StandardScaler()
    X_inv_sc = sc_x.fit_transform(X_inv)
    y_inv_sc = sc_y.fit_transform(y_inv)

    m_inv = MLPRegressor(hidden_layer_sizes=(128,128,64,32), max_iter=2000, random_state=42)
    m_inv.fit(X_inv_sc, y_inv_sc)

    return m_freq, m_bw, m_gain, scaler, m_inv, sc_x, sc_y, df

with st.spinner("Loading CAEAD models..."):
    m_freq, m_bw, m_gain, scaler, m_inv, sc_x, sc_y, df = load_and_train()

st.success("✅ All models loaded!")

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Forward Design",
    "🎯 Inverse Design",
    "📊 Dataset Explorer"
])

# ════════════════════════════════════════════════
# TAB 1 — Forward Design
# ════════════════════════════════════════════════
with tab1:
    st.subheader("Forward Design — Geometry → Performance")
    st.markdown("*Antenna dimensions daalo — AI performance predict karega*")

    col1, col2 = st.columns(2)
    with col1:
        length = st.slider("Patch Length (mm)", 10.0, 40.0, 20.0, 0.5)
        width  = st.slider("Patch Width (mm)",  10.0, 40.0, 20.0, 0.5)
        er     = st.slider("Substrate Er",       2.2,  10.2,  4.4, 0.1)
    with col2:
        height = st.slider("Substrate Height (mm)", 0.8, 3.2, 1.6, 0.1)
        feed   = st.slider("Feed Position",          0.1, 0.4, 0.2, 0.01)
        ground = st.slider("Ground Ratio",           1.2, 2.0, 1.5, 0.05)

    if st.button("⚡ Predict Performance", type="primary"):
        inp = np.array([[length, width, er, height, feed, ground]])
        inp_sc = scaler.transform(inp)

        freq = m_freq.predict(inp_sc)[0]
        bw   = m_bw.predict(inp_sc)[0]
        gain = m_gain.predict(inp_sc)[0]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("📡 Resonant Frequency", f"{freq:.3f} GHz")
        c2.metric("📶 Bandwidth",          f"{bw:.2f} MHz")
        c3.metric("📈 Gain",               f"{gain:.2f} dBi")

        # S11 plot
        f_range = np.linspace(1e9, 10e9, 500)
        f0 = freq * 1e9
        BW_hz = bw * 1e6
        s11 = 20 * np.log10(
            (f_range - f0)**2 / ((f_range - f0)**2 + (BW_hz/2)**2 + 1e-10)
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(f_range/1e9, s11, color='blue', linewidth=2)
        ax.axhline(-10, color='red', linestyle='--', label='-10 dB')
        ax.axvline(freq, color='green', linestyle='--', label=f'{freq:.2f} GHz')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('S11 (dB)')
        ax.set_title(f'Predicted S11 — {freq:.2f} GHz | {bw:.1f} MHz BW | {gain:.1f} dBi')
        ax.legend(); ax.grid(True); ax.set_ylim(-40, 5)
        st.pyplot(fig)

        st.info("⚡ Traditional HFSS: 2-8 hours | CAEAD: < 1 second")

# ════════════════════════════════════════════════
# TAB 2 — Inverse Design
# ════════════════════════════════════════════════
with tab2:
    st.subheader("Inverse Design — Performance → Geometry")
    st.markdown("*Target specs daalo — AI best antenna geometry batayega*")

    col1, col2, col3 = st.columns(3)
    with col1:
        t_freq = st.slider("Target Frequency (GHz)", 1.0, 10.0, 2.4, 0.1)
    with col2:
        t_bw   = st.slider("Target Bandwidth (MHz)", 1.0, 60.0, 5.0, 0.5)
    with col3:
        t_gain = st.slider("Target Gain (dBi)", 0.0, 6.0, 2.0, 0.1)

    # Quick presets
    st.markdown("**Quick Presets:**")
    p1, p2, p3 = st.columns(3)
    
    if st.button("📶 WiFi 2.4 GHz", use_container_width=False):
        t_freq, t_bw, t_gain = 2.4, 5.0, 2.0
    if st.button("📶 WiFi 5 GHz", use_container_width=False):
        t_freq, t_bw, t_gain = 5.0, 10.0, 4.0
    if st.button("📱 5G Sub-6", use_container_width=False):
        t_freq, t_bw, t_gain = 3.5, 20.0, 3.5

    if st.button("🎯 Generate Antenna Design", type="primary"):
        inp = np.array([[t_freq, t_bw, t_gain]])
        inp_sc = sc_x.transform(inp)
        out_sc = m_inv.predict(inp_sc)
        out    = sc_y.inverse_transform(out_sc)[0]

        out[0] = np.clip(out[0], 10, 40)
        out[1] = np.clip(out[1], 10, 40)
        out[2] = np.clip(out[2], 2.2, 10.2)
        out[3] = np.clip(out[3], 0.8, 3.2)

        st.divider()
        st.success(f"✅ Antenna design generated for {t_freq} GHz!")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📏 Patch Length", f"{out[0]:.2f} mm")
        c2.metric("📐 Patch Width",  f"{out[1]:.2f} mm")
        c3.metric("🔬 Substrate Er", f"{out[2]:.2f}")
        c4.metric("📦 Height",       f"{out[3]:.2f} mm")

        st.markdown("**Suggested Substrate:**")
        if out[2] < 3.0:
            st.info("Rogers RT/duroid 5880 (Er=2.2) — Low loss, aerospace grade")
        elif out[2] < 5.0:
            st.info("Rogers RO4003C (Er=3.55) — Low loss, high frequency")
        elif out[2] < 7.0:
            st.info("FR4 (Er=4.4) — Standard PCB, cost effective")
        else:
            st.info("Rogers RO3010 (Er=10.2) — High Er, compact design")

        st.info("⚡ Traditional HFSS design cycle: 2-8 hours | CAEAD: < 1 second")

# ════════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ════════════════════════════════════════════════
with tab3:
    st.subheader("Dataset Explorer — 5000 Antenna Designs")

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X Axis", df.columns[1:], index=0)
    with col2:
        y_axis = st.selectbox("Y Axis", df.columns[1:], index=6)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df[x_axis], df[y_axis], alpha=0.3, s=5, color='steelblue')
    ax.set_xlabel(x_axis); ax.set_ylabel(y_axis)
    ax.set_title(f'{x_axis} vs {y_axis} — 5000 Designs')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Total: {len(df)} designs | Parameters: {len(df.columns)-1}")

st.divider()
st.caption("CAEAD — Center for AI-Driven Electromagnetics & Antenna Design | Dr. B. Pritam Singh | IIT (ISM) Dhanbad | github.com/pritamsingh-antenna/caead2025")