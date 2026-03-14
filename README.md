# 📡 CAEAD — Center for AI-Driven Electromagnetics & Antenna Design

**Founder:** Dr. B. Pritam Singh  
**Ph.D.** RF & Microwave Engineering | IIT (ISM) Dhanbad  
**IEEE Member** #92711308 | 14 SCI/SCOPUS Papers | 484 Citations | h-index: 10

---

## 🎯 What is CAEAD?

India's first AI-driven antenna design automation system.  
**Target:** Replace 2-8 hour HFSS simulations with AI predictions in under 1 second.

---

## ✅ Results Achieved

| Metric | Value |
|--------|-------|
| Dataset | 500 antenna designs |
| DNN Accuracy | R² = 0.99 (99%) |
| Frequency Error | < 0.8% |
| Design Time | < 1 second vs 2-8 hours |
| Speedup | ~10,000x faster than HFSS |

---

## 🧠 AI Pipeline
```
Target Frequency (GHz)
        ↓
DNN Surrogate Model (64-64-32)
        ↓
Bayesian Optimizer (200 iterations)
        ↓
Best Antenna Design (Length, Width, Er, Height)
        ↓
Predicted Frequency within 0.8% error
```

## 📁 Files

| File | Description |
|------|-------------|
| `s11_basic.py` | S11 simulation — single patch antenna |
| `antenna_dataset.py` | 20 antenna designs dataset |
| `caead_ml_model.py` | DNN surrogate model training |
| `caead_optimizer.py` | Bayesian optimization engine |
| `caead_dashboard.py` | Interactive Streamlit dashboard |
| `caead_500_designs.csv` | 500-point simulation dataset |

---

## 🚀 Run Dashboard Locally
```bash
git clone https://github.com/pritamsingh-antenna/caead2025.git
cd caead2025
pip install streamlit scikit-learn pandas numpy matplotlib
streamlit run caead_dashboard.py
```

---

## 🔭 Next Steps

- [ ] PyAEDT integration — real HFSS simulation data
- [ ] 50,000 design dataset via Latin Hypercube Sampling
- [ ] PINN — Physics-Informed Neural Network
- [ ] cVAE Inverse Design
- [ ] IEEE TAP/APS paper submission
- [ ] Indian Patent Office filing

---

*CAEAD — Building the future of electromagnetic design with AI*
