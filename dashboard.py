import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="AI Cybersecurity Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Load alert data
# -----------------------------

alerts_path = Path("outputs/alerts.csv")

if not alerts_path.exists():
    st.error("outputs/alerts.csv was not found. Run the ML pipeline first.")
    st.stop()

df = pd.read_csv(alerts_path)

# -----------------------------
# Dashboard title
# -----------------------------

st.title("🛡️ AI-Powered Cybersecurity Threat Detection")
st.caption("Machine Learning Threat Detection Dashboard")

st.divider()

# -----------------------------
# Summary metrics
# -----------------------------

total_records = len(df)

threats = int((df["prediction"] == 1).sum())
normal = int((df["prediction"] == 0).sum())

threat_rate = (threats / total_records * 100) if total_records else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{total_records:,}")
col2.metric("Threats Detected", f"{threats:,}")
col3.metric("Normal Records", f"{normal:,}")
col4.metric("Threat Rate", f"{threat_rate:.2f}%")

st.divider()

# -----------------------------
# Charts
# -----------------------------

left, right = st.columns(2)

with left:
    st.subheader("Prediction Distribution")

    prediction_counts = (
        df["prediction"]
        .map({0: "Normal", 1: "Threat"})
        .value_counts()
    )

    st.bar_chart(prediction_counts)

with right:
    st.subheader("Threat Probability")

    st.line_chart(df["threat_probability"].head(100))

st.divider()

# -----------------------------
# Recent alerts
# -----------------------------

st.subheader("🚨 Recent Detection Results")

display_columns = [
    "label",
    "prediction",
    "threat_probability",
    "alert"
]

available_columns = [
    column for column in display_columns
    if column in df.columns
]

recent = df[available_columns].head(20).copy()

if "label" in recent.columns:
    recent["label"] = recent["label"].map({
        0: "Normal",
        1: "Threat"
    })

if "prediction" in recent.columns:
    recent["prediction"] = recent["prediction"].map({
        0: "Normal",
        1: "Threat"
    })

if "threat_probability" in recent.columns:
    recent["threat_probability"] = recent["threat_probability"].round(4)

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.caption(
    "This dashboard is an educational machine-learning prototype "
    "using the NSL-KDD dataset. It does not monitor live network traffic."
)
