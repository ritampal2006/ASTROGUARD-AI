# 🛰️ AstroGuard AI: AI-Powered Space Debris Detection & Collision Alert System

**AstroGuard AI** is a production-grade, full-stack aerospace telemetry and orbital collision-risk mitigation platform. Inspired by next-generation Space Operations Centers (SOC) and planetary defense mission command dashboards, AstroGuard AI combines real orbital tracking telemetry (Hugging Face Space-Track TLE data) with a high-performance **Random Forest Classifier** (`scikit-learn`) to predict orbital conjunction threats, calculate required evasive delta-V maneuvers, and visualize satellite orbits in real-time.

---

## 🌟 Key Features

### 🚀 1. Conjunction Analyzer (Main Demo)
- **Multi-variable Conjunction Telemetry:** Interactively simulate miss distances ($0.1$–$50.0\text{ km}$), closing velocities ($1.0$–$15.0\text{ km/s}$), time to closest approach / TCA ($0.1$–$120.0\text{ min}$), and orbital inclination delta ($0.0^\circ$–$45.0^\circ$).
- **One-Click Mission Presets:** Test real-world conjunction scenarios (*ISS vs Cosmos-2251 Debris*, *Starlink vs Fengyun-1C*, *Sentinel-6A Proximity*, and *GEO Nominal Sweep*).
- **Dynamic Threat Alert Banner:** Real-time visual feedback flashing **Green** (Low Risk / Routine), **Amber** (Medium Risk / Conjunction Data Message Stage 2), or **Glowing Crimson Strobe** (High Risk / Immediate Delta-V Evasive Burn).
- **Dynamic 10-Point Trajectory Chart (`Chart.js`):** Plots approaching debris distance against escalating collision probability over time.

### 🌍 2. Live Orbit Visualizer (HTML5 Canvas Radar)
- **Top-Down Holographic Orbit Plane:** Center 3D-feel Earth with glowing atmospheric halo and orbital shells (LEO, SSO, MEO, GEO).
- **Real-Time Orbit Mechanics:** Real-time tracking of friendly assets (*ISS Alpha*, *Tiangong CSS*, *Starlink-4210*, *Sentinel-6A*, *GPS Block III*) and hazardous debris fragments (*Cosmos-2251*, *Fengyun-1C*, rocket upper stages).
- **Target Threat Conjunction Vectors:** Pulsing red targeting lines and danger cones highlighting active close-approach trajectories.

### 📊 3. Space Debris Insights
- **NORAD / ESA Space Debris Telemetry:** Analysis extracted from the Hugging Face `juliensimon/space-track-tle-history` dataset ($50,000+$ sample).
- **Interactive Visualizations:** Orbital distribution doughnut chart (LEO $74.2\%$, GEO $17.3\%$, MEO $8.5\%$), closing velocity spectrum histogram, and inclination dispersion chart.
- **Live Conjunction Telemetry Log:** Real-time feed of recent conjunction events and active flight maneuvers.

### 🤖 4. Model Diagnostics & Explainability
- **Random Forest Architecture:** $120$ Estimator Trees, max depth $14$, trained on stratified encounter distributions with $99.9\%+$ test accuracy and zero false-negatives on critical high-risk collisions.
- **Gini Feature Importances:** Visual horizontal bar chart illustrating the primary decision drivers (Miss Distance $\approx 84.7\%$, TCA $\approx 14.0\%$, Relative Speed $\approx 1.2\%$, Inclination $\approx 0.1\%$).
- **Full Evaluation Confusion Matrix & Technical Specs.**

---

## 🗂️ Project File Structure

```
AstroGuard AI/
├── app.py                     # Flask backend API & inference server
├── train_model.py             # Dataset engineering, Hugging Face TLE loader & model trainer
├── astroguard_model.pkl       # Serialized Random Forest model (scikit-learn)
├── model_metadata.json        # Evaluation metrics, confusion matrix & feature weights
├── templates/
│   └── index.html             # Complete Sci-Fi Command Center UI (HTML5, Glassmorphism CSS, Canvas, Chart.js)
├── test_suite.py              # In-process Flask API unit test suite
└── README.md                  # System documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
py -m pip install flask scikit-learn joblib pandas numpy datasets pyarrow
```

### 2. (Optional) Re-train the Model
To re-train the Random Forest classifier and refresh `astroguard_model.pkl`:
```bash
py train_model.py
```

### 3. Launch the Application
```bash
py app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📡 Backend API Endpoints

### 1. `POST /predict`
Evaluates orbital encounter parameters and returns real-time risk classification:
```json
// Request Body
{
  "distance": 0.85,
  "relative_speed": 12.4,
  "time_to_approach": 8.5,
  "inclination_diff": 28.4
}

// Response
{
  "status": "success",
  "risk_level": "High",
  "risk_code": 2,
  "collision_probability_pct": 92.4,
  "probabilities": {
    "Low": 0.0,
    "Medium": 0.0,
    "High": 100.0
  },
  "protocol": {
    "title": "CRITICAL COLLISION THREAT",
    "action": "IMMEDIATE EVASIVE ACTION MANDATORY: Execute retrograde/radial thruster burn to elevate perigee.",
    "urgency": "RED // EMERGENCY MANEUVER",
    "maneuver_required": true
  },
  "telemetry": {
    "estimated_delta_v_ms": 3.55,
    "kinetic_energy_mj": 384.4,
    "maneuver_decision_window_min": 5.1
  }
}
```

### 2. `POST /simulate-trajectory`
Generates a 10-point time-series projection from initial detection down to closest approach.

### 3. `GET /api/stats`
Retrieves model metrics, feature importances, and Space-Track TLE telemetry statistics.
