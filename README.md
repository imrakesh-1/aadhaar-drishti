# Aadhaar Drishti 👁️
**AI-Driven Operational Intelligence for UIDAI Governance**

**Participant**: Rakesh Mondal (IIT Madras) | **Team ID**: UIDAI_7034

Aadhaar Drishti is an interactive command center designed to optimize Indian district-level operations through automated anomaly detection and predictive resource allocation.

## 🚀 Key Features
- **Pulse Monitor**: Real-time visibility into national enrolment and update trends with automated deduplication of 100+ misspelled districts.
- **Anomaly Hunter**: Built-in `IsolationForest` (Unsupervised ML) to detect "hotspots" and operational irregularities.
- **Infrastructure Allocator**: Predictive logic to calculate Biometric Kit and Staff requirements based on projected growth.

## 📈 Data Insights & Stats 
*(Extracted from live dashboard analysis)*

### 🕵️ Anomaly Diagnostics
- **Districts Scanned**: 905
- **Anomalies Detected**: 46
- **Max Risk Score**: 0.23
- **Highest Risk Districts**:
  - **Thane (Maharashtra)**: Risk Score 0.228 | Load: ~1M+
  - **Pune (Maharashtra)**: Risk Score 0.226 | Load: ~1M+
  - **North West Delhi (Delhi)**: Risk Score 0.208
  - **Bengaluru (Karnataka)**: Risk Score 0.196

### 📊 Seasonal Trends
- **The July Surge**: Data confirms a **300% spike** in demographic updates every July, correlating with national school admission cycles.
- **Biometric Maintenance**: Biometric update volume now matches or exceeds enrolment volume in saturated states, signaling a shift from "Acquisition" to "Maintenance."

### 🏗️ Resource Allocation (Example)
- **District**: Andaman (Andaman & Nicobar Islands)
- **Current Annual Load**: 2,765
- **Projected Monthly Load (at +20% growth)**: 3,318
- **Required**: 3 Biometric Kits & 3 Dedicated Staff.

## 🛠️ Tech Stack
- **Backend**: Python (AadhaarBrain Engine)
- **ML**: Scikit-Learn (Isolation Forest)
- **Visualization**: Plotly & Streamlit

## ⚙️ How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Dashboard
python3 -m streamlit run app.py
```

---
*Developed for the Aadhaar Hackathon 2026*
