# Emergency Department Flow Optimisation
## A Process Mining & Predictive Analytics Case Study

**Analyst:** Zhongyi (Betty) Hu  
**Program:** Master of Management Analytics, Rotman School of Management  
**Dataset:** 16,011 ED visits · 90,965 clinical events · April–June 2021  
**Tools:** Python · Streamlit · scikit-learn · Plotly  
**Interactive Dashboard:** [github.com/HzyBetty/emergency-department-operations-analytics](https://github.com/HzyBetty/emergency-department-operations-analytics)

---

## Executive Summary

Emergency Departments operate under simultaneous pressure from rising patient volumes, constrained bed capacity, and strict clinical safety benchmarks. Yet the event-level data that could drive operational decisions — timestamped records of every triage, assessment, consult, and discharge — is rarely used in real time.

This project applies **process mining** to extract the actual patient pathway from raw event logs, quantifies where time is lost at each care transition, and deploys a **predictive model** to flag high admission-risk patients before bed demand peaks. The analysis covers 16,011 ED visits across a two-month period and is delivered as a live, interactive analytics dashboard.

**Three headline findings drive the recommendations:**

| Finding | Implication |
|---|---|
| Consult-triggered visits account for 17% of volume but 9.35 hrs median LOS — 3.5× the non-consult baseline | Consult response time is the single largest driver of extended stays and downstream bed blockage |
| The Assessment → Discharge transition averages 128 minutes across 11,648 visits | Post-assessment discharge processing is a systemic delay affecting the majority of patients |
| A Random Forest model predicts admission with AUC-ROC = 0.946; pathway complexity (step count) is the dominant predictor | High-complexity patients can be identified for early bed booking before clinical confirmation of admission |

A targeted 20% reduction in post-assessment discharge processing time — achievable through discharge lounge protocols and nurse-led discharge — would save an estimated **0.64 hrs per patient** and reduce extended stays (>8 hrs) from 15.2% to approximately 10%, corresponding to ~$172,000 in weekly operational savings at conservative cost assumptions.

---

## 1. Data & Methodology

### 1.1 Dataset

The event log contains **90,965 timestamped clinical events** across **16,011 unique ED visits** between 31 March and 1 June 2021. Each row records a single activity (e.g., Triage, Assessment, Consult Request) for one patient visit, along with patient demographics, triage acuity, assigned zone, and disposition outcome.

**Event types captured:**

| Event | Role in Pathway |
|---|---|
| Ambulance Arrival / Ambulance Transfer | Patient arrival via paramedic |
| Registration | Administrative intake |
| Triage | Acuity assessment (CTAS Level 1–5) |
| Assessment | Physician evaluation |
| Consult Request / Consult Arrival | Specialist involvement |
| Discharge / Left ED | Visit end |

**Triage distribution:**

| Level | Category | Visits | Share |
|---|---|---|---|
| L1 | Resuscitation | 134 | 0.8% |
| L2 | Emergent | 4,822 | 30.1% |
| L3 | Urgent | 8,737 | 54.6% |
| L4 | Less Urgent | 1,947 | 12.2% |
| L5 | Non-Urgent | 367 | 2.3% |

### 1.2 Data Engineering

**Visit identification.** The source dataset includes a `visit_id` field used as the primary visit key throughout the analysis. No timestamp-based visit reconstruction was required.

**Clinical KPI derivation.** Three performance metrics were computed from event timestamps:

- **Length of Stay (LOS):** Time from a visit's first recorded event to its last
- **Door-to-Triage time:** Time from arrival (Ambulance Arrival or Registration) to the Triage event
- **Door-to-Physician time:** Time from arrival to the first Assessment event

**LWBS detection.** A visit is classified as Left Without Being Seen (LWBS) if a "Left ED" event occurs before any "Assessment" event in the chronological sequence. 130 visits (0.8%) meet this criterion.

**Step duration.** Within each visit, the elapsed time between consecutive events is calculated as the per-visit time difference. These step durations form the basis of bottleneck analysis.

### 1.3 Analytical Pipeline

```
Raw Event Log (CSV)
       │
       ▼
Data Cleaning & Feature Engineering
  · Deduplication on (visit_id, timestamp, event)
  · CTAS triage standardisation (numeric code → labelled level)
  · Admission flag from disposition code (code 7 = inpatient)
  · LOS, Door-to-Triage, Door-to-Physician, LWBS per visit
       │
       ├─▶ Process Flow Analysis
       │     Directly-follows transitions → Sankey diagram
       │     Step duration aggregation → bottleneck ranking
       │     CTAS benchmark comparison → compliance gap
       │
       ├─▶ Predictive Modelling
       │     Features: age, triage level, zone, gender,
       │                arrival hour, day of week, step count
       │     Model: Random Forest (200 trees, max depth 8)
       │     Validation: 80/20 stratified split, AUC-ROC
       │
       └─▶ Simulation
             Bootstrap resample from actual LOS distribution
             → LOS shift under targeted efficiency gains
             → Extended stay reduction · financial impact
```

### 1.4 Design Decisions

**One-hot encoding over ordinal encoding.** Triage levels and zones are treated as nominal categories using `pd.get_dummies`. Ordinal encoding would impose an artificial linear relationship between category distances, distorting feature importance in tree-based models.

**Median over mean for LOS benchmarking.** The LOS distribution is right-skewed (mean 5.02 hrs vs. median 3.22 hrs), driven by a long tail of extended stays. Median provides a more robust operational benchmark; mean is reported separately to capture tail-risk exposure.

**Bootstrap simulation over parametric simulation.** Scenario modelling draws 2,000 resampled values from the observed LOS distribution rather than fitting a normal distribution. This preserves the actual skew and bimodality of ED length-of-stay data.

---

## 2. Key Findings

### 2.1 Overall Operational Performance

The ED is currently operating within acceptable parameters on most headline metrics, with a median LOS of **3.22 hours** — below the 4-hour threshold that triggers elevated operational status. However, mean LOS of **5.02 hours** reveals significant tail-risk exposure.

| Metric | Value | Benchmark / Context |
|---|---|---|
| Total visits (2-month period) | 16,011 | — |
| Median Length of Stay | 3.22 hrs | < 4 hr target |
| Mean Length of Stay | 5.02 hrs | 56% above median — skewed by long stays |
| Extended stays > 8 hrs | **15.2%** of visits | 1 in 7 patients |
| 90th percentile LOS | 10.75 hrs | Tail-risk benchmark |
| Admission rate | 13.9% | — |
| LWBS rate | **0.8%** | < 2% clinical safety threshold |
| Median Door-to-Triage | 7.0 min | CTAS L2 benchmark: 15 min ✅ |
| Median Door-to-Physician | 38.0 min | — |

**So what:** The gap between median (3.22 hrs) and mean (5.02 hrs) LOS signals that the system-level average is being pulled upward by a concentrated group of long-stay patients — primarily consult cases and high-acuity admissions. Operational improvement efforts should target this tail specifically.

---

### 2.2 Patient Pathway — Process Map & Bottlenecks

The Sankey diagram below (reproduced from the interactive dashboard) maps the volume-weighted patient pathway. The main highway is:

**Ambulance Arrival / Registration → Triage → Assessment → Discharge**

Two distinct sub-pathways branch from Assessment:

- **Consult pathway (17% of visits):** Assessment → Consult Request → Consult Arrival → Discharge or Left ED
- **Direct discharge (83% of visits):** Assessment → Discharge

**Step-level delay analysis — top transitions by average wait:**

| Rank | Transition | Avg Wait (Min) | 90th Pct (Min) | Volume |
|---|---|---|---|---|
| 1 | Consult Arrival → Left ED | 289 | 753 | 978 |
| 2 | Consult Request → Left ED | 218 | 465 | 150 |
| 3 | **Assessment → Consult Request** | **174** | **341** | **2,628** |
| 4 | Consult Request → Discharge | 163 | 327 | 1,082 |
| 5 | Assessment → Left ED | 153 | 349 | 1,413 |
| 6 | **Consult Request → Consult Arrival** | **139** | **294** | **1,468** |
| 7 | **Assessment → Discharge** | **128** | **273** | **11,648** |
| 8 | Registration → Left ED | 96 | 192 | 121 |

**So what:** Three transitions stand out:
1. The **consult initiation wait** (Assessment → Consult Request, avg 174 min) suggests physician decision-making delay after assessment — a potential target for standardised escalation criteria.
2. The **consult response time** (Consult Request → Consult Arrival, avg 139 min) is a specialist availability problem — a rostering and paging-protocol issue rather than an ED problem.
3. The **Assessment → Discharge** transition (avg 128 min, 11,648 visits) is the highest-volume bottleneck in the entire system. Even a modest improvement here has the largest aggregate impact.

---

### 2.3 Protocol Compliance — CTAS Triage Benchmarks

The Canadian Triage and Acuity Scale (CTAS) specifies maximum door-to-triage times by acuity level.

| Triage Level | Median D-to-T (Actual) | CTAS Benchmark | Status |
|---|---|---|---|
| L1 – Resuscitation | 7.0 min | 0 min (immediate) | ❌ Fails |
| L2 – Emergent | 7.0 min | 15 min | ✅ Passes |
| L3 – Urgent | 7.0 min | 30 min | ✅ Passes |
| L4 – Less Urgent | 7.0 min | 60 min | ✅ Passes |
| L5 – Non-Urgent | 20.0 min | 120 min | ✅ Passes |

**So what:** Four of five triage levels meet CTAS benchmarks comfortably. The critical gap is **L1 Resuscitation** — the benchmark requires immediate assessment (0 minutes), yet the median observed time is 7 minutes. Given that L1 patients have a **59% admission rate** and a median LOS of **6.15 hours**, any delay in this group carries the highest clinical risk of the entire patient population. A dedicated resuscitation intake protocol — bypassing the shared triage queue — is the indicated intervention.

---

### 2.4 Acuity-Stratified Performance

**Length of Stay by triage level:**

| Level | Median LOS | Mean LOS | 90th Pct LOS | Admission Rate |
|---|---|---|---|---|
| L1 – Resuscitation | 6.15 hrs | 9.18 hrs | 21.74 hrs | 59.0% |
| L2 – Emergent | 4.72 hrs | 6.97 hrs | 15.28 hrs | 24.9% |
| L3 – Urgent | 3.02 hrs | 4.61 hrs | 9.58 hrs | 10.5% |
| L4 – Less Urgent | 1.67 hrs | 2.31 hrs | 4.09 hrs | 1.4% |
| L5 – Non-Urgent | 1.22 hrs | 1.77 hrs | 3.11 hrs | 0.5% |

**LWBS rate by triage level:**

| Level | LWBS Rate | Clinical Interpretation |
|---|---|---|
| L1 – Resuscitation | 0.0% | No LWBS — appropriate for this acuity |
| L2 – Emergent | 0.7% | Within threshold |
| L3 – Urgent | 0.8% | Within threshold |
| **L4 – Less Urgent** | **1.3%** | Highest rate — patients leaving during long waits |
| L5 – Non-Urgent | 0.8% | Within threshold |

**So what:** L4 patients have the highest LWBS rate, likely reflecting frustration with waiting times that feel disproportionate to perceived urgency. Fast-track streaming for L4–L5 patients — handled by nurse practitioners or physician assistants rather than the main physician pool — is the standard intervention. This also frees physician capacity for L1–L3 cases.

---

### 2.5 Physical Zone Load & Efficiency

| Zone | Visits | Median LOS | Admission Rate | Interpretation |
|---|---|---|---|---|
| YZ | 4,232 | 3.72 hrs | 9% | High-volume general zone — moderate efficiency |
| GZ | 4,112 | 1.67 hrs | 1% | Fast-track stream — high throughput, low complexity |
| A | 3,193 | 5.73 hrs | 31% | High-acuity, high-volume — primary bed-pressure zone |
| EPZ | 2,503 | 2.15 hrs | 4% | Efficient moderate-acuity stream |
| SA | 1,078 | 7.55 hrs | 41% | Highest LOS zone — specialist-intensive cases |
| Resus | 421 | 6.58 hrs | 55% | Resuscitation zone — expected high complexity |

**So what:** **Zone A** is the most operationally significant target: it handles 3,193 visits with a 31% admission rate and 5.73 hr median LOS. This zone's combination of high volume and high admission rate makes it the primary driver of inpatient bed demand. Dedicated bed-ahead coordination between Zone A nurses and inpatient wards would reduce the time from admission decision to physical transfer.

**Zone SA** shows the highest median LOS (7.55 hrs) and admission rate (41%), consistent with complex specialist cases. These patients generate the majority of consult traffic and are the key cohort contributing to the mean-vs-median LOS gap noted in Section 2.1.

---

### 2.6 Admission Prediction Model

A **Random Forest classifier** was trained on six patient and pathway features to predict inpatient admission.

**Model performance:**

| Metric | Value |
|---|---|
| AUC-ROC | **0.946** |
| Training set | 12,809 visits (80%) |
| Test set | 3,202 visits (20%) |
| Stratified split | Yes (preserves 13.9% base rate) |

**Feature importance — top predictors:**

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | **Pathway Complexity** (step count) | 0.616 | Number of clinical events in the visit |
| 2 | Patient Age | 0.117 | Older patients significantly more likely to be admitted |
| 3 | Zone SA | 0.044 | Specialist-assessment zone — strong admission signal |
| 4 | Zone A | 0.040 | High-acuity general zone |
| 5 | Zone GZ | 0.036 | Fast-track zone — negative signal (low admission) |
| 6 | Resuscitation zone | 0.034 | Expected high-admission signal |
| 7 | L2 Emergent triage | 0.030 | Acuity-based admission signal |
| 8 | Arrival hour | 0.019 | Late-night arrivals skew toward admission |

**So what:** The dominance of **pathway complexity** (step count, importance = 0.616) is the most operationally significant finding from the model. It means that by the time a patient has passed through their 4th or 5th clinical event, the model can predict admission with high confidence — before any physician explicitly orders a bed. This supports a **proactive bed request protocol**: flag any L1–L2 patient with 4+ steps for immediate bed coordination, allowing inpatient wards 4–6 hours of preparation time.

The AUC-ROC of **0.946** places this model well above the 0.80 threshold considered clinically useful for triage support tools, and significantly above the 0.75 operational utility threshold.

---

### 2.7 Consult Pathway Analysis

Specialist consults are triggered in **2,718 visits (17.0% of all encounters)**.

| Metric | Consult Visits | Non-Consult Visits |
|---|---|---|
| Median LOS | **9.35 hrs** | 2.68 hrs |
| Admission rate | **74.9%** | ~5% |
| LOS multiplier vs. non-consult | **3.5×** | — |

**So what:** A consult request is the strongest real-world predictor of both admission and extended LOS, yet consult-triggered cases account for the largest bottleneck wait times in the system (Assessment → Consult Request: 174 min avg). The 139-minute average consult response time (Consult Request → Consult Arrival) represents a specialist-availability and paging-protocol problem. A time-to-response SLA for on-call specialists — with an escalation pathway after 60 minutes — would directly compress the median LOS of this high-risk cohort.

---

## 3. Recommendations & Business Impact

The following recommendations are prioritised by the combination of **impact magnitude** (how many patients are affected) and **implementation complexity** (how operationally feasible the intervention is).

---

### Priority 1 — Discharge Lounge & Nurse-Led Discharge Protocol
**Target bottleneck:** Assessment → Discharge (avg 128 min, 11,648 visits — 73% of all visits)  
**Rationale:** This is the highest-volume transition in the system. Post-assessment delays typically arise from prescription processing, patient education, and transport coordination — activities that do not require physician presence.

**Intervention:** Introduce a discharge lounge staffed by a registered nurse. Patients cleared for discharge by the physician move to the lounge, freeing the ED bed for the next patient.

**Projected impact (20% reduction in post-assessment discharge time):**

| Metric | Baseline | Target |
|---|---|---|
| Median LOS | 3.22 hrs | 2.57 hrs |
| Extended stays > 8 hrs | 15.2% | ~10.2% |
| Estimated hrs saved / week | — | ~$172,000 weekly operational savings |

---

### Priority 2 — Consult Response SLA & Escalation Protocol
**Target bottleneck:** Consult Request → Consult Arrival (avg 139 min, 1,468 occurrences)  
**Rationale:** Consult visits have a 9.35 hr median LOS — 3.5× the non-consult baseline. A 139-minute average specialist response represents the largest per-patient delay in the system.

**Intervention:** Establish a 60-minute SLA for specialist response to ED consult requests, with an automatic escalation to the on-call specialist supervisor at 75 minutes. Track compliance monthly using the consult response time metric available in the event log.

**Projected impact:** A 50% reduction in consult response time (from 139 to ~70 min) across 1,468 consult visits per 2-month period would recover approximately **101,000 patient-minutes** of ED bed occupancy, enabling meaningful throughput improvement in Zone SA and Zone A — the two highest-LOS zones.

---

### Priority 3 — Proactive Bed Request Protocol (Enabled by ML Model)
**Target metric:** Admission rate 13.9%; pathway complexity dominates admission prediction (feature importance: 0.616)  
**Rationale:** The ML model achieves AUC-ROC = 0.946, meaning high-risk patients can be identified with high confidence from pathway data alone — before physician confirmation of admission.

**Intervention:** Implement an automated flag in the ED patient tracking system: any L1 or L2 patient whose step count reaches 4 receives an automatic bed request to the inpatient allocation desk, with a 4-hour provisional hold. The model's per-visit admission probability (outputted in the dashboard's "High-Risk Visit Flag List") provides the decision support.

**Projected impact:** Currently, bed-request-to-physical-transfer delays contribute to extended LOS in the 13.9% of visits that result in admission. A 2-hour reduction in this transfer gap — achievable through proactive coordination — would reduce the median LOS of admitted patients by an estimated 30–40 minutes.

---

### Priority 4 — L1 Resuscitation Dedicated Intake Lane
**Target bottleneck:** L1 door-to-triage = 7 min (benchmark: 0 min)  
**Rationale:** L1 patients have a 59% admission rate and 6.15 hr median LOS. Any triage queue delay for this cohort is both a clinical safety and accreditation risk.

**Intervention:** Designate the Resus zone with a direct intake pathway that bypasses the shared triage queue. L1 patients identified by paramedics are escorted directly to Resus, with triage and assessment conducted concurrently by the resuscitation team.

**Projected impact:** Elimination of the 7-minute median triage delay for L1 patients removes a measurable clinical risk for 134 patients in the 2-month observation period — approximately 800 patients per year.

---

### Priority 5 — Fast-Track Stream for L4–L5 Patients
**Target metric:** L4 LWBS rate 1.3% (highest of all triage levels); L4–L5 median LOS 1.67 hrs and 1.22 hrs respectively  
**Rationale:** L4 and L5 patients represent 14.5% of volume, have low acuity, and are clinically suitable for management by nurse practitioners or physician assistants. Keeping them in the main physician queue increases their wait time and elevates their LWBS rate — the only group above 1%.

**Intervention:** A separate fast-track stream, staffed by advanced practice providers, handles all L4–L5 patients independently of the main ED flow. This reduces main-stream congestion and brings LWBS toward the 0% target for non-urgent cases.

---

## 4. Methodological Notes

**On the simulation model.** Scenario projections use bootstrap resampling from the observed LOS distribution rather than parametric assumptions. This preserves the right-skew and bimodality of actual ED LOS data, making projections more conservative and realistic than normal-distribution-based models.

**On the ML model.** The Random Forest model is trained and evaluated on the same two-month dataset. Generalisation to different periods, patient populations, or ED configurations would require re-training and re-validation. The model is intended as a **decision support signal**, not a deterministic clinical decision tool.

**On financial estimates.** The $130/hr/patient operational cost assumption is a conservative estimate based on published Canadian ED cost-per-hour figures. Actual figures vary by institution, staffing model, and cost-accounting methodology.

**On CTAS benchmarks.** Door-to-triage benchmarks are drawn from the Canadian Triage and Acuity Scale (2020 revision). The same acuity categories are used globally under different naming conventions (e.g., ESI in the United States), and the underlying time targets are broadly comparable.

---

## Appendix — Technical Summary

| Component | Specification |
|---|---|
| Language | Python 3.9+ |
| App framework | Streamlit 1.31+ |
| ML model | RandomForestClassifier (n=200, max_depth=8, min_samples_leaf=5) |
| Validation | Stratified 80/20 train-test split |
| Model metric | AUC-ROC = 0.946 |
| Encoding | pd.get_dummies (one-hot, no ordinal assumptions) |
| Simulation | Bootstrap resample, n=2,000, seed=42 |
| Process map | Plotly Sankey, fixed node arrangement, forward-flow filtered |
| Bottleneck detection | Statistical: Mean + 2 SD threshold |
| Benchmarks | CTAS (Canadian Triage and Acuity Scale) |
| Data privacy | Dataset is anonymised; no PHI included |
