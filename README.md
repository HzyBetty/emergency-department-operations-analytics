# ED Flow Intelligence — Emergency Department Process Mining & Analytics Platform

An interactive analytics dashboard that transforms raw ED event logs into actionable operational intelligence. Built with Python and Streamlit, it applies **process mining**, **machine learning**, and **stochastic simulation** to help healthcare operations teams reduce patient Length of Stay (LOS), improve protocol compliance, and forecast inpatient demand.

---

## Business Problem

Emergency Departments generate high-volume, noisy event logs that are rarely used for real-time operational decisions. This platform bridges that gap by converting raw timestamped activity data into three categories of insight:

| Category | Output |
|---|---|
| **Process visibility** | Where patients go, how long each step takes, and where flow breaks down |
| **Risk stratification** | Which patients are likely to be admitted — before bed demand peaks |
| **Capacity planning** | How targeted efficiency improvements shift LOS distributions and reduce operational cost |

---

## Key Features

### 📊 Executive Overview
- Five operational KPIs calculated from raw event data: **Total Visits, Median LOS, Admission Rate, LWBS Rate, and Median Door-to-Triage Time**
- System status banner (Optimal / Elevated / Critical) driven by median LOS thresholds
- LOS distribution by triage level (box plot) — shows acuity-specific variance, not just averages
- Admission and LWBS rates broken down by triage level for targeted prioritisation
- Visit volume by day of week for staffing rotation planning

### 🔀 Patient Flow & Bottlenecks
- **Sankey diagram**: maps actual patient pathways with link width proportional to volume. Red branches to "Left ED" directly visualise LWBS volume.
- **Step-level delay ranking**: top transitions sorted by average wait time, with 90th-percentile column to capture tail-risk delays
- **Door-to-Triage vs. CTAS Benchmarks**: median performance overlaid against Canadian Triage and Acuity Scale targets. Green = compliant, Red = accreditation gap.

### 🤖 Risk & Admission Analytics
- **Random Forest classifier** trained on six patient and pathway features. Reports AUC-ROC for transparency.
- **Feature importance chart**: identifies which factors most strongly predict admission (triage level, pathway complexity, age, arrival hour, zone, gender)
- **High-risk visit flag list**: patients with predicted admission probability ≥ 70%, enabling proactive bed booking 4–6 hours ahead
- **Zone load vs. LOS efficiency**: volume and median LOS by physical zone — targets for nurse redeployment

### 📈 Capacity Planning & Alerts
- **Hourly congestion heatmap** (top 12 transitions): distinguishes systemic process failures from shift-change handoff delays
- **Scenario simulation**: bootstrap-resampled from actual LOS data. Projects how a target efficiency gain shifts the distribution and reduces extended stays (> 8 hrs). Includes financial impact estimate.
- **Statistical anomaly detection**: flags visits exceeding Mean + 2 SD on any single wait step, with a per-visit Root Cause Analysis timeline

---

## Technical Architecture

```
data/event_log_ED_MMA_2026.csv
         │
         ▼
load_and_preprocess()          # deduplication, triage mapping, admission flag,
         │                     # step duration, time features
         ▼
compute_visit_metrics()        # visit-level KPIs: LOS, door-to-triage,
         │                     # door-to-physician, LWBS detection
         ▼
Streamlit 4-tab UI             # filters → tab renders → plotly charts
         │
         ├── Sankey / Bottleneck (networkx-free — built from transition counts)
         ├── ML pipeline (pd.get_dummies → RandomForest → roc_auc_score)
         └── Bootstrap simulation (np.random.choice from real LOS data)
```

**Key design decisions:**
- Uses `visit_id` from the source data as the primary visit key (no fragile timestamp-based reconstruction)
- `pd.get_dummies` instead of `LabelEncoder` — avoids ordinal-coding error in tree-based models
- `@st.cache_data` on both preprocessing functions — data only re-loaded when the file changes
- Bootstrap resampling for simulation preserves the real shape of the LOS distribution

---

## Data Schema

| Column | Type | Description |
|---|---|---|
| `visit_id` | string | Unique ED visit identifier |
| `patient_id` | string | Patient identifier |
| `timestamp` | datetime | Event timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `event` | string | Clinical step (e.g., Triage, Assessment, Discharge) |
| `triage_code` | int (1–5) | Acuity level per CTAS |
| `initial_zone` | string | Physical ED zone assigned at arrival |
| `disposition_code` | int | Outcome code (7 = inpatient admit) |
| `disposition_desc` | string | Outcome description |
| `age` | int | Patient age |
| `gender` | string | Patient gender |

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- `pip`

### Setup

```bash
# Clone the repository
git clone https://github.com/HzyBetty/project_process_mining.git
cd project_process_mining

# (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The app auto-loads `data/event_log_ED_MMA_2026.csv` on startup. Use the sidebar upload widget to override with a different dataset.

---

## Triage Benchmark Reference (CTAS)

| Level | Category | Door-to-Triage Target |
|---|---|---|
| L1 | Resuscitation | Immediate (0 min) |
| L2 | Emergent | ≤ 15 min |
| L3 | Urgent | ≤ 30 min |
| L4 | Less Urgent | ≤ 60 min |
| L5 | Non-Urgent | ≤ 120 min |

---

## Data Privacy

The bundled dataset (`data/`) is anonymised and contains no Protected Health Information (PHI). All patient identifiers are synthetic. The tool is designed for demonstration and portfolio purposes.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| App framework | Streamlit 1.31+ |
| Data processing | pandas, numpy |
| Visualisation | Plotly (Express + Graph Objects) |
| Machine learning | scikit-learn (RandomForest, AUC-ROC) |
| Simulation | NumPy bootstrap resampling |
