import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="ED Flow Intelligence",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded",
)

# ============================================================================
# GLOBAL STYLES
# ============================================================================
st.markdown(
    """
<style>
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 12px 16px;
    }
    .insight-box {
        background: #e8f4fd;
        border-left: 4px solid #0066cc;
        padding: 10px 14px;
        border-radius: 0 4px 4px 0;
        font-size: 0.88em;
        color: #084298;
        margin: 10px 0 14px 0;
    }
    .status-critical { background:#f8d7da; border-left:5px solid #dc3545; padding:12px 18px; border-radius:4px; margin-bottom:16px; }
    .status-warning  { background:#fff3cd; border-left:5px solid #fd7e14; padding:12px 18px; border-radius:4px; margin-bottom:16px; }
    .status-ok       { background:#d1e7dd; border-left:5px solid #198754; padding:12px 18px; border-radius:4px; margin-bottom:16px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# DATA ENGINE
# ============================================================================
DEFAULT_DATA_PATH = "data/event_log_ED_MMA_2026.csv"

TRIAGE_MAP = {
    1: "L1 – Resuscitation",
    2: "L2 – Emergent",
    3: "L3 – Urgent",
    4: "L4 – Less Urgent",
    5: "L5 – Non-Urgent",
}
TRIAGE_ORDER = list(TRIAGE_MAP.values())

# CTAS door-to-triage benchmarks (minutes)
CTAS_DTT = {
    "L1 – Resuscitation": 0,
    "L2 – Emergent": 15,
    "L3 – Urgent": 30,
    "L4 – Less Urgent": 60,
    "L5 – Non-Urgent": 120,
}


@st.cache_data
def load_and_preprocess(file_source):
    """Load, clean, and engineer base features from the ED event log."""
    df = pd.read_csv(file_source if isinstance(file_source, str) else file_source)

    # ── Core cleanup ──────────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["visit_id"] = df["visit_id"].astype(str)
    df = df.drop_duplicates(subset=["visit_id", "timestamp", "event"])
    df = df.sort_values(["visit_id", "timestamp"]).reset_index(drop=True)

    # ── Triage standardisation ────────────────────────────────────────────────
    df["triage_code"] = pd.to_numeric(df["triage_code"], errors="coerce")
    df["triage_label"] = df["triage_code"].map(TRIAGE_MAP).fillna("Unknown")

    # ── Admission flag ────────────────────────────────────────────────────────
    # disposition_code 7 = inpatient admit in this dataset; text fallback added
    df["is_admitted"] = (
        (pd.to_numeric(df["disposition_code"], errors="coerce") == 7)
        | df["disposition_desc"].str.contains("Admit|admit", na=False)
    ).astype(int)

    # ── Time features ─────────────────────────────────────────────────────────
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()

    # ── Intra-visit step duration (minutes) ───────────────────────────────────
    df["step_duration_min"] = (
        df.groupby("visit_id")["timestamp"].diff().dt.total_seconds() / 60
    )

    return df


@st.cache_data
def compute_visit_metrics(df):
    """Build a visit-level summary DataFrame with clinical KPIs."""

    def _first(s):
        v = s.dropna()
        return v.iloc[0] if len(v) > 0 else np.nan

    # ── Base aggregation ──────────────────────────────────────────────────────
    visit = (
        df.groupby("visit_id")
        .agg(
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
            triage_label=("triage_label", _first),
            triage_code=("triage_code", _first),
            age=("age", _first),
            gender=("gender", _first),
            initial_zone=("initial_zone", _first),
            is_admitted=("is_admitted", "max"),
            hour_of_day=("hour_of_day", "first"),
            day_of_week=("day_of_week", "first"),
            step_count=("event", "count"),
        )
        .reset_index()
    )
    visit["los_hours"] = (
        (visit["last_ts"] - visit["first_ts"]).dt.total_seconds() / 3600
    )

    # ── Door-to-Triage time ───────────────────────────────────────────────────
    triage_ts = (
        df[df["event"] == "Triage"].groupby("visit_id")["timestamp"].min().rename("triage_ts")
    )
    arrival_ts = (
        df[df["event"].isin(["Ambulance Arrival", "Registration"])]
        .groupby("visit_id")["timestamp"]
        .min()
        .rename("arrival_ts")
    )
    t1 = triage_ts.to_frame().join(arrival_ts)
    t1["door_to_triage_min"] = (t1["triage_ts"] - t1["arrival_ts"]).dt.total_seconds() / 60
    t1 = t1[t1["door_to_triage_min"].between(0, 480)]  # sanity: 0–8 h
    visit = visit.merge(t1[["door_to_triage_min"]], on="visit_id", how="left")

    # ── Door-to-Physician time ────────────────────────────────────────────────
    assess_ts = (
        df[df["event"] == "Assessment"].groupby("visit_id")["timestamp"].min().rename("assess_ts")
    )
    t2 = assess_ts.to_frame().join(arrival_ts)
    t2["door_to_physician_min"] = (t2["assess_ts"] - t2["arrival_ts"]).dt.total_seconds() / 60
    t2 = t2[t2["door_to_physician_min"].between(0, 720)]  # sanity: 0–12 h
    visit = visit.merge(t2[["door_to_physician_min"]], on="visit_id", how="left")

    # ── LWBS detection ────────────────────────────────────────────────────────
    # "Left Without Being Seen" = patient left ED before any Assessment event
    ordered_events = (
        df.sort_values("timestamp")
        .groupby("visit_id")["event"]
        .apply(list)
        .rename("events_list")
    )
    visit = visit.merge(ordered_events, on="visit_id", how="left")

    def _is_lwbs(evts):
        if not isinstance(evts, list) or "Left ED" not in evts:
            return False
        left_idx = evts.index("Left ED")
        return "Assessment" not in evts[:left_idx]

    visit["is_lwbs"] = visit["events_list"].apply(_is_lwbs)

    return visit


# ============================================================================
# SIDEBAR & DATA LOADING
# ============================================================================
st.sidebar.markdown("### 🏥 ED Flow Intelligence")
st.sidebar.markdown("---")

# ── Data source ───────────────────────────────────────────────────────────────
if os.path.exists(DEFAULT_DATA_PATH):
    use_bundled = st.sidebar.checkbox("Use bundled dataset", value=True)
    if use_bundled:
        file_source = DEFAULT_DATA_PATH
    else:
        file_source = st.sidebar.file_uploader("Upload Event Log (CSV)", type=["csv"])
else:
    file_source = st.sidebar.file_uploader("Upload Event Log (CSV)", type=["csv"])

if file_source is None:
    st.title("ED Flow Intelligence")
    st.markdown("**Emergency Department Process Mining & Performance Analytics**")
    st.info(
        "Place your event log CSV in the `/data` folder, or upload it via the sidebar to begin."
    )
    st.stop()

with st.spinner("Processing data…"):
    df = load_and_preprocess(file_source)
    visit_df = compute_visit_metrics(df)

# ── Triage filter ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
available_triage = [t for t in TRIAGE_ORDER if t in df["triage_label"].unique()]
selected_triage = st.sidebar.multiselect(
    "Triage Level", available_triage, default=available_triage
)

df_f = df[df["triage_label"].isin(selected_triage)].copy()
visit_f = visit_df[visit_df["triage_label"].isin(selected_triage)].copy()

# ── Transition table ──────────────────────────────────────────────────────────
df_f["prev_event"] = df_f.groupby("visit_id")["event"].shift(1)
trans_df = df_f.dropna(subset=["prev_event"]).copy()
trans_df = trans_df[trans_df["event"] != trans_df["prev_event"]]
trans_df["transition"] = trans_df["prev_event"] + " → " + trans_df["event"]

# ============================================================================
# SUMMARY METRICS
# ============================================================================
total_visits = len(visit_f)
median_los = visit_f["los_hours"].median()
admission_rate = visit_f["is_admitted"].mean()
lwbs_rate = visit_f["is_lwbs"].mean()
median_dtt = visit_f["door_to_triage_min"].median()

if median_los > 6:
    status_cls = "status-critical"
    status_msg = "⚠️ CRITICAL — Median LOS exceeds 6 hrs. Escalate staffing and consider diversion."
elif median_los > 4:
    status_cls = "status-warning"
    status_msg = "🟡 ELEVATED — Median LOS 4–6 hrs. Activate flow protocols and review bed capacity."
else:
    status_cls = "status-ok"
    status_msg = "✅ OPTIMAL — Median LOS within operational targets."

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊  Executive Overview",
        "🔀  Patient Flow & Bottlenecks",
        "🤖  Risk & Admission Analytics",
        "📈  Capacity Planning & Alerts",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ·  EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f'<div class="{status_cls}"><b>SYSTEM STATUS:</b> {status_msg}</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Visits", f"{total_visits:,}")
    c2.metric("Median LOS", f"{median_los:.1f} hrs")
    c3.metric("Admission Rate", f"{admission_rate:.1%}")
    c4.metric(
        "LWBS Rate",
        f"{lwbs_rate:.1%}",
        help="Left Without Being Seen — patients who departed before physician assessment.",
    )
    c5.metric("Median Door-to-Triage", f"{median_dtt:.0f} min")
    st.markdown("---")

    # Row 1: LOS distribution + day-of-week volume
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("#### Length of Stay by Triage Level")
        los_plot = visit_f[visit_f["los_hours"].between(0, 24)]
        fig_los = px.box(
            los_plot,
            x="triage_label",
            y="los_hours",
            color="triage_label",
            category_orders={"triage_label": TRIAGE_ORDER},
            color_discrete_sequence=["#d73027", "#f46d43", "#fdae61", "#74add1", "#313695"],
            labels={"triage_label": "", "los_hours": "Length of Stay (Hours)"},
        )
        fig_los.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig_los, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 L1–L2 patients show the widest LOS variance — '
            "extended stays in these acuity groups are the primary driver of downstream inpatient "
            "bed blockage and hallway care.</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("#### Visit Volume by Day of Week")
        DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = visit_f["day_of_week"].value_counts().reindex(DOW_ORDER, fill_value=0)
        fig_dow = px.bar(
            x=dow_counts.index,
            y=dow_counts.values,
            color=dow_counts.values,
            color_continuous_scale="Blues",
            labels={"x": "", "y": "Visit Count"},
        )
        fig_dow.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=340,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 Weekend surges warrant enhanced staffing rotations '
            "and extended on-call physician coverage to maintain CTAS response benchmarks.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Row 2: Admission rate + LWBS rate by triage level
    st.markdown("#### Admission & LWBS Rates by Triage Level")
    triage_summary = (
        visit_f.groupby("triage_label")
        .agg(visits=("visit_id", "count"), admitted=("is_admitted", "sum"), lwbs=("is_lwbs", "sum"))
        .reset_index()
    )
    triage_summary["Admission Rate"] = triage_summary["admitted"] / triage_summary["visits"]
    triage_summary["LWBS Rate"] = triage_summary["lwbs"] / triage_summary["visits"]
    triage_summary = (
        triage_summary[triage_summary["triage_label"].isin(TRIAGE_ORDER)]
        .set_index("triage_label")
        .reindex(TRIAGE_ORDER)
        .reset_index()
    )

    col_c, col_d = st.columns(2)
    with col_c:
        fig_adm = px.bar(
            triage_summary,
            x="triage_label",
            y="Admission Rate",
            color="Admission Rate",
            color_continuous_scale="Reds",
            text=triage_summary["Admission Rate"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else ""
            ),
            labels={"triage_label": ""},
        )
        fig_adm.update_traces(textposition="outside")
        fig_adm.update_layout(
            showlegend=False, coloraxis_showscale=False, height=300, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_adm, use_container_width=True)

    with col_d:
        fig_lwbs = px.bar(
            triage_summary,
            x="triage_label",
            y="LWBS Rate",
            color="LWBS Rate",
            color_continuous_scale="Oranges",
            text=triage_summary["LWBS Rate"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else ""
            ),
            labels={"triage_label": ""},
        )
        fig_lwbs.update_traces(textposition="outside")
        fig_lwbs.update_layout(
            showlegend=False, coloraxis_showscale=False, height=300, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_lwbs, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 LWBS rates above 2% represent a patient safety risk '
        "and directly reduce revenue. The triage level with the highest LWBS rate is the "
        "priority target for wait-time reduction initiatives.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ·  PATIENT FLOW & BOTTLENECKS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    # ── Sankey diagram ────────────────────────────────────────────────────────
    st.markdown("### Patient Pathway Analysis")
    st.caption(
        "Link width is proportional to patient volume. "
        "Red branches indicate patients who left before physician assessment (LWBS)."
    )

    min_vol = st.slider("Minimum pathway volume to display", 1, 200, 20, key="sankey_vol")
    trans_counts = trans_df["transition"].value_counts()
    trans_counts = trans_counts[trans_counts >= min_vol]

    # Canonical step order — used to filter backward transitions that cause loops
    STEP_ORDER = {
        "Ambulance Arrival": 0, "Ambulance Transfer": 0,
        "Triage": 1, "Registration": 1,   # parallel entry points; same level
        "Assessment": 2,
        "Consult Request": 3, "Consult Arrival": 4,
        "CDU Transfer": 5,
        "Discharge": 6, "Left ED": 6,
    }
    # Fixed x positions (left → right reflects clinical time sequence)
    X_MAP = {
        "Ambulance Arrival":  0.02, "Ambulance Transfer": 0.02,
        "Triage":             0.25, "Registration":       0.25,
        "Assessment":         0.52,
        "Consult Request":    0.68, "Consult Arrival":    0.78,
        "CDU Transfer":       0.87,
        "Discharge":          0.97, "Left ED":            0.97,
    }
    # Fixed y positions to separate co-located nodes vertically
    Y_MAP = {
        "Ambulance Transfer": 0.12,
        "Ambulance Arrival":  0.32,
        "Registration":       0.52,
        "Triage":             0.72,
        "Assessment":         0.45,
        "Consult Request":    0.25,
        "Consult Arrival":    0.20,
        "CDU Transfer":       0.12,
        "Discharge":          0.32,
        "Left ED":            0.72,
    }
    NODE_COLORS = {
        "Ambulance Arrival":  "#4e79a7", "Ambulance Transfer": "#4e79a7",
        "Triage":             "#59a14f", "Registration":       "#59a14f",
        "Assessment":         "#f28e2b",
        "Consult Request":    "#b07aa1", "Consult Arrival":    "#b07aa1",
        "CDU Transfer":       "#9c755f",
        "Discharge":          "#76b7b2", "Left ED":            "#e15759",
    }

    if len(trans_counts) > 0:
        # Build node list (preferred order first, unknowns appended)
        PREFERRED_ORDER = list(X_MAP.keys())
        all_nodes = list(dict.fromkeys([n for t in trans_counts.index for n in t.split(" → ")]))
        ordered_nodes = [n for n in PREFERRED_ORDER if n in all_nodes]
        ordered_nodes += [n for n in all_nodes if n not in ordered_nodes]
        node_idx = {n: i for i, n in enumerate(ordered_nodes)}

        # Assign fixed positions; unknown nodes default to centre
        x_vals = [X_MAP.get(n, 0.50) for n in ordered_nodes]
        y_vals = [Y_MAP.get(n, 0.50) for n in ordered_nodes]

        sources, targets, values, link_colors = [], [], [], []
        for trans, count in trans_counts.items():
            parts = trans.split(" → ")
            if len(parts) != 2:
                continue
            src, tgt = parts
            if src not in node_idx or tgt not in node_idx:
                continue
            # Forward-flow filter: skip transitions that go more than 1 step backward
            # (they create the looping bands seen with snap arrangement)
            src_step = STEP_ORDER.get(src, 99)
            tgt_step = STEP_ORDER.get(tgt, 99)
            if tgt_step < src_step - 1:
                continue
            sources.append(node_idx[src])
            targets.append(node_idx[tgt])
            values.append(int(count))
            link_colors.append(
                "rgba(220, 53, 69, 0.35)" if tgt == "Left ED"
                else "rgba(158, 202, 225, 0.40)"
            )

        fig_sankey = go.Figure(
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=18,
                    thickness=22,
                    line=dict(color="white", width=0.5),
                    label=ordered_nodes,
                    color=[NODE_COLORS.get(n, "#aaa") for n in ordered_nodes],
                    x=x_vals,
                    y=y_vals,
                ),
                link=dict(source=sources, target=targets, value=values, color=link_colors),
            )
        )
        fig_sankey.update_layout(
            height=540,
            margin=dict(t=20, b=20, l=10, r=10),
            font=dict(size=13, color="#333"),
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 Nodes are arranged left-to-right by clinical time sequence. '
            "The widest links define your department's <b>main highway</b>. "
            "Narrowing flows mark capacity pinch points. "
            "Red branches to 'Left ED' directly quantify LWBS patients — each one a patient "
            "safety event and lost revenue.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No transitions meet the minimum volume threshold. Lower the slider.")

    st.markdown("---")

    # ── Bottleneck bar chart ──────────────────────────────────────────────────
    st.markdown("### Step-Level Delay Analysis")
    st.caption("Average wait time between consecutive care steps, ranked by severity of delay.")

    bn = (
        trans_df.groupby("transition")
        .agg(
            avg_wait=("step_duration_min", "mean"),
            median_wait=("step_duration_min", "median"),
            p90_wait=("step_duration_min", lambda x: x.quantile(0.9)),
            volume=("transition", "count"),
        )
        .reset_index()
        .sort_values("avg_wait", ascending=False)
    )
    bn = bn[bn["volume"] >= 5]

    fig_bn = px.bar(
        bn.head(12),
        x="avg_wait",
        y="transition",
        orientation="h",
        color="avg_wait",
        color_continuous_scale="YlOrRd",
        text=bn.head(12)["avg_wait"].round(1),
        labels={"avg_wait": "Avg Wait (Min)", "transition": ""},
    )
    fig_bn.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    fig_bn.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        height=430,
        margin=dict(t=10, b=10, r=80),
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_bn, use_container_width=True)

    with st.expander("Full bottleneck detail — all transitions"):
        st.dataframe(
            bn.rename(
                columns={
                    "transition": "Care Step Transition",
                    "avg_wait": "Avg Wait (Min)",
                    "median_wait": "Median (Min)",
                    "p90_wait": "90th Pct (Min)",
                    "volume": "Volume",
                }
            ).round(1),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Door-to-Triage vs CTAS benchmarks ────────────────────────────────────
    st.markdown("### Door-to-Triage Performance vs. CTAS Benchmarks")
    st.caption("Median door-to-triage time per triage level. Green = meets CTAS standard. Red = below standard.")

    dtt = (
        visit_f.groupby("triage_label")["door_to_triage_min"]
        .median()
        .reset_index()
        .rename(columns={"door_to_triage_min": "Actual Median (Min)"})
    )
    dtt["CTAS Target (Min)"] = dtt["triage_label"].map(CTAS_DTT)
    dtt["meets_target"] = dtt["Actual Median (Min)"] <= dtt["CTAS Target (Min)"]
    dtt = dtt.dropna(subset=["Actual Median (Min)"])

    fig_dtt = go.Figure()
    fig_dtt.add_trace(
        go.Bar(
            x=dtt["triage_label"],
            y=dtt["Actual Median (Min)"],
            name="Actual Median",
            marker_color=["#28a745" if m else "#dc3545" for m in dtt["meets_target"]],
            text=dtt["Actual Median (Min)"].round(1),
            texttemplate="%{text:.1f} min",
            textposition="outside",
        )
    )
    fig_dtt.add_trace(
        go.Scatter(
            x=dtt["triage_label"],
            y=dtt["CTAS Target (Min)"],
            mode="markers+lines",
            name="CTAS Benchmark",
            marker=dict(symbol="diamond", size=12, color="#0066cc"),
            line=dict(dash="dash", color="#0066cc", width=2),
        )
    )
    fig_dtt.update_layout(
        height=360,
        margin=dict(t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Minutes",
        xaxis_title="",
    )
    st.plotly_chart(fig_dtt, use_container_width=True)
    st.markdown(
        '<div class="insight-box">💡 Each red bar is a documented accreditation and patient safety gap. '
        "Triage levels failing their CTAS benchmark should be the first target for fast-track stream "
        "redesign or dedicated intake nursing roles.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ·  RISK & ADMISSION ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Admission Risk Profiling")
    st.caption(
        "A Random Forest model identifies the key predictors of inpatient admission "
        "from patient demographics, triage parameters, and pathway complexity."
    )

    # ── ML pipeline ───────────────────────────────────────────────────────────
    ml_df = visit_f.copy()
    CAT_COLS = ["gender", "triage_label", "initial_zone", "day_of_week"]
    NUM_COLS = ["age", "hour_of_day", "step_count"]

    feature_df = pd.get_dummies(
        ml_df[CAT_COLS + NUM_COLS], columns=CAT_COLS, drop_first=False
    ).fillna(0)

    target = ml_df["is_admitted"]
    valid_mask = target.notna() & ml_df["age"].notna()
    X = feature_df[valid_mask]
    y = target[valid_mask]

    ml_ready = y.nunique() > 1 and len(y) >= 50

    if ml_ready:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
        )
        rf.fit(X_train, y_train)
        y_prob_test = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob_test)

        # Feature importance — cleaned labels
        feat_imp = (
            pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(12)
        )
        feat_imp["Feature"] = (
            feat_imp["Feature"]
            .str.replace("triage_label_", "Triage: ", regex=False)
            .str.replace("initial_zone_", "Zone: ", regex=False)
            .str.replace("gender_", "Gender: ", regex=False)
            .str.replace("day_of_week_", "Day: ", regex=False)
            .str.replace("step_count", "Pathway Complexity", regex=False)
            .str.replace("hour_of_day", "Arrival Hour", regex=False)
            .str.replace("age", "Patient Age", regex=False)
        )
        feat_imp = feat_imp.sort_values("Importance", ascending=True)

        ml_col1, ml_col2 = st.columns([3, 2])

        with ml_col1:
            st.markdown("#### Top Admission Predictors")
            fig_imp = px.bar(
                feat_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Teal",
                labels={"Importance": "Relative Importance", "Feature": ""},
            )
            fig_imp.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                height=400,
                margin=dict(t=10, b=10, r=20),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with ml_col2:
            st.markdown("#### Model Performance")
            auc_delta = (
                f"{auc - 0.75:+.3f} vs. 0.75 threshold"
                if pd.notna(auc)
                else "N/A"
            )
            st.metric(
                "AUC-ROC",
                f"{auc:.3f}",
                delta=auc_delta,
                help="0.5 = random baseline, 1.0 = perfect. Above 0.75 is operationally useful.",
            )
            st.markdown("---")
            st.markdown(
                """
**Clinical Application:**
- Flag visits with > 60% admission probability for early bed booking
- Notify inpatient wards 4–6 hrs ahead of anticipated transfers
- Prioritise IV access, labs, and imaging for high-probability patients
"""
            )

        st.markdown(
            '<div class="insight-box">💡 If "Pathway Complexity" (step count) ranks in the '
            "top three, it confirms that diagnostic workup depth is a strong leading indicator "
            "for admission — supporting early bed request protocols triggered at the 3rd or 4th "
            "care step.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── High-risk visit flag list ─────────────────────────────────────────
        st.markdown("### High-Risk Visit Flag List")
        st.caption(
            "Visits with predicted admission probability ≥ 70% — "
            "recommend early inpatient bed booking request."
        )

        all_features = feature_df[valid_mask].reindex(columns=X.columns, fill_value=0)
        ml_df_valid = ml_df[valid_mask].copy()
        ml_df_valid["admission_prob"] = rf.predict_proba(all_features)[:, 1]

        high_risk = (
            ml_df_valid[ml_df_valid["admission_prob"] >= 0.70]
            .sort_values("admission_prob", ascending=False)[
                ["visit_id", "triage_label", "age", "gender", "initial_zone",
                 "admission_prob", "step_count", "is_admitted"]
            ]
            .copy()
        )
        high_risk["admission_prob"] = high_risk["admission_prob"].apply(lambda x: f"{x:.0%}")

        if not high_risk.empty:
            st.dataframe(
                high_risk.rename(
                    columns={
                        "visit_id": "Visit ID",
                        "triage_label": "Triage",
                        "initial_zone": "Zone",
                        "admission_prob": "Predicted Admission Prob.",
                        "step_count": "Complexity Score",
                        "is_admitted": "Actually Admitted",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"{len(high_risk):,} visits ({len(high_risk)/total_visits:.1%} of volume) "
                "flagged as high-risk for admission."
            )
        else:
            st.success(
                "No visits exceed the 70% admission probability threshold with current filters."
            )

    else:
        st.warning(
            "Insufficient data for ML model. Ensure the dataset has varied admission outcomes "
            "and at least 50 visits after filtering."
        )

    st.markdown("---")

    # ── Zone load & efficiency ─────────────────────────────────────────────────
    st.markdown("### Physical Zone Load & Efficiency")
    z1, z2 = st.columns(2)

    with z1:
        st.markdown("#### Visit Volume by Zone")
        zc = visit_f["initial_zone"].value_counts().reset_index()
        zc.columns = ["Zone", "Visits"]
        fig_z1 = px.bar(
            zc, x="Zone", y="Visits", color="Visits",
            color_continuous_scale="Blues", text="Visits"
        )
        fig_z1.update_traces(textposition="outside")
        fig_z1.update_layout(
            showlegend=False, coloraxis_showscale=False, height=320, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_z1, use_container_width=True)

    with z2:
        st.markdown("#### Median LOS by Zone (Hours)")
        zlos = visit_f.groupby("initial_zone")["los_hours"].median().reset_index()
        zlos.columns = ["Zone", "Median LOS (Hours)"]
        fig_z2 = px.bar(
            zlos, x="Zone", y="Median LOS (Hours)",
            color="Median LOS (Hours)", color_continuous_scale="Oranges",
            text="Median LOS (Hours)"
        )
        fig_z2.update_traces(texttemplate="%{text:.1f} h", textposition="outside")
        fig_z2.update_layout(
            showlegend=False, coloraxis_showscale=False, height=320, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_z2, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 A zone with high visit volume <i>and</i> high median LOS '
        "is the primary target for nurse redeployment or stream redesign. Volume alone without "
        "LOS context can be misleading.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  ·  CAPACITY PLANNING & ALERTS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    # ── Hourly congestion heatmap ─────────────────────────────────────────────
    st.markdown("### Hourly Congestion Heatmap")
    st.caption(
        "Average wait time (minutes) per care-step transition across all 24 hours. "
        "Focus on persistent dark horizontal bands — they indicate systemic delays "
        "independent of time-of-day."
    )

    top_trans = trans_df["transition"].value_counts().head(12).index.tolist()
    hm_data = (
        trans_df[trans_df["transition"].isin(top_trans)]
        .groupby(["hour_of_day", "transition"])["step_duration_min"]
        .mean()
        .unstack()
        .fillna(0)
    )
    fig_heat = px.imshow(
        hm_data.T,
        color_continuous_scale="YlOrRd",
        labels={"x": "Hour of Day", "y": "", "color": "Avg Wait (Min)"},
        aspect="auto",
    )
    fig_heat.update_layout(height=430, margin=dict(t=10, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(
        '<div class="insight-box">💡 Persistent horizontal bands indicate structural bottlenecks. '
        "Diagonal concentrations around shift-change hours (7am, 3pm, 11pm) signal handoff delays "
        "— a target for structured SBAR communication protocols.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Efficiency simulation ─────────────────────────────────────────────────
    st.markdown("### Scenario Simulation: LOS Reduction Impact")
    st.caption(
        "Bootstrap-resampled from actual LOS distribution — the shift in the curve reflects "
        "realistic operational improvement rather than a theoretical normal distribution."
    )

    sim_col1, sim_col2 = st.columns([3, 2])
    with sim_col1:
        eff_gain = st.select_slider(
            "Target Efficiency Gain (%)",
            options=[5, 10, 15, 20, 25, 30, 40, 50],
            value=20,
        )
        actual_los = visit_f["los_hours"].dropna()
        actual_los = actual_los[actual_los.between(0.1, 24)]

        np.random.seed(42)
        baseline_sim = np.random.choice(actual_los, size=2000, replace=True)
        improved_sim = baseline_sim * (1 - eff_gain / 100)

        sim_plot = pd.DataFrame(
            {
                "LOS (Hours)": np.concatenate([baseline_sim, improved_sim]),
                "Scenario": ["Current Baseline"] * 2000
                + [f"{eff_gain}% Improvement Target"] * 2000,
            }
        )
        bm = float(np.median(baseline_sim))
        im = float(np.median(improved_sim))

        fig_sim = px.histogram(
            sim_plot,
            x="LOS (Hours)",
            color="Scenario",
            barmode="overlay",
            nbins=60,
            opacity=0.75,
            color_discrete_map={
                "Current Baseline": "#636EFA",
                f"{eff_gain}% Improvement Target": "#00CC96",
            },
        )
        fig_sim.add_vline(
            x=bm, line_dash="dash", line_color="#636EFA",
            annotation_text=f"Baseline: {bm:.1f} h", annotation_position="top right",
        )
        fig_sim.add_vline(
            x=im, line_dash="dash", line_color="#00CC96",
            annotation_text=f"Target: {im:.1f} h", annotation_position="top left",
        )
        fig_sim.update_layout(
            height=370,
            margin=dict(t=20, b=10),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    with sim_col2:
        st.markdown("#### Projected Impact")
        hrs_saved = bm - im
        tail_base = (baseline_sim > 8).mean()
        tail_imp = (improved_sim > 8).mean()
        weekly_vol = total_visits / max(visit_f["day_of_week"].nunique(), 1) * 7

        st.metric("Hrs Saved per Patient", f"{hrs_saved:.2f} hrs")
        st.metric("Extended Stays > 8 h — Baseline", f"{tail_base:.1%}")
        st.metric(
            "Extended Stays > 8 h — Target",
            f"{tail_imp:.1%}",
            delta=f"{tail_imp - tail_base:.1%}",
        )
        st.metric("Est. Hours Saved / Week", f"{hrs_saved * weekly_vol:,.0f} hrs")

        cost_per_hr = 130  # conservative ED operational cost estimate (USD/hr)
        weekly_savings = hrs_saved * weekly_vol * cost_per_hr
        st.markdown(
            f"""
---
**Financial Estimate:**
At ≈${cost_per_hr}/hr/patient, a {eff_gain}% LOS reduction across ≈{weekly_vol:.0f}
weekly visits corresponds to **≈${weekly_savings:,.0f} saved per week** in operational costs.
"""
        )

    st.markdown("---")

    # ── Statistical anomaly detection ─────────────────────────────────────────
    st.markdown("### Statistical Anomaly Detection — Red Flag Visits")
    st.caption(
        "Visits where any single wait step exceeds Mean + 2 SD are flagged for clinical review. "
        "These represent outlier process failures: missed handoffs, overcrowding events, or "
        "documentation gaps."
    )

    step_mean = trans_df["step_duration_min"].mean()
    step_std = trans_df["step_duration_min"].std()
    threshold = step_mean + 2 * step_std
    anomalies = trans_df[trans_df["step_duration_min"] > threshold]

    if not anomalies.empty:
        rf_count = anomalies["visit_id"].nunique()
        st.metric(
            "Red Flag Visits Detected",
            f"{rf_count:,}",
            f"{rf_count / total_visits:.1%} of total volume",
        )
        selected_visit = st.selectbox(
            "Select visit for Root Cause Analysis timeline:",
            sorted(anomalies["visit_id"].unique()),
        )
        vd = (
            df_f[df_f["visit_id"] == selected_visit]
            .sort_values("timestamp")[["timestamp", "event", "step_duration_min"]]
            .copy()
        )
        vd["step_duration_min"] = vd["step_duration_min"].round(1)
        vd["timestamp"] = vd["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        vd.columns = ["Timestamp", "Clinical Event", "Wait Since Prior Step (Min)"]
        st.dataframe(vd, use_container_width=True, hide_index=True)
    else:
        st.success(
            "All visit transitions are within statistical control limits (Mean + 2 SD)."
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
<div style="color:#999;font-size:0.78em;text-align:center;padding:8px 0;">
ED Flow Intelligence Platform &nbsp;|&nbsp;
Process Mining · Predictive Analytics · Operational Simulation &nbsp;|&nbsp;
Built with Python, Streamlit, scikit-learn & Plotly &nbsp;|&nbsp;
Triage benchmarks per CTAS (Canadian Triage and Acuity Scale) guidelines
</div>
""",
        unsafe_allow_html=True,
    )
