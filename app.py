import streamlit as st

from src.predictor import predict_job
from src.risk_score import (
    analyze_job,
    calculate_risk_score,
    get_risk_level,
    get_safety_tips
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Intelligent Fake Job Posting Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📋 Project Information")

st.sidebar.markdown("""
### Intelligent Fake Job Posting Detection

### 👨‍💻 Technologies Used
- Apache Spark
- Python
- TF-IDF
- Logistic Regression
- Streamlit
- Rule-Based Risk Scoring

### 📂 Dataset
Fake Job Postings Dataset (Kaggle)

### 📊 Dataset Statistics
- Total Jobs : 17,880
- Real Jobs : 17,014
- Fake Jobs : 866

### 🎯 Features
- Fake/Real Job Prediction
- Fraud Risk Score
- Suspicious Indicator Detection
- Safety Recommendations
""")

# --------------------------------------------------
# Main Heading
# --------------------------------------------------
st.title("🛡️ Intelligent Fake Job Posting Detection")
st.subheader("Fraud Risk Scoring System using Big Data Analytics")

st.write(
    "Paste the complete job posting below and click **Analyze Job Posting**."
)

st.divider()

# --------------------------------------------------
# Single Text Area
# --------------------------------------------------
st.subheader("📝 Paste Complete Job Posting")

job_text = st.text_area(
    "",
    placeholder="""
Example:

Software Engineer

Company: Google

Job Description:
Google is looking for a Software Engineer to develop scalable backend systems.

Requirements:
• Bachelor's Degree in Computer Science
• Python
• SQL
• Problem Solving

Benefits:
• Health Insurance
• Paid Leave
• Flexible Working Hours
""",
    height=350
)

# --------------------------------------------------
# Analyze Button
# --------------------------------------------------
if st.button("🚀 Analyze Job Posting", use_container_width=True):

    if not job_text.strip():
        st.warning("Please paste a job posting.")
        st.stop()

    # Machine Learning Prediction
    prediction, probability = predict_job(job_text)

    # Rule-Based Analysis
    _, indicators = analyze_job(job_text)

    # Final Risk Score
    risk_score = calculate_risk_score(
        probability,
        indicators
    )

    risk_level = get_risk_level(risk_score)

    st.divider()

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        if prediction == 1:
            st.error("🚨 Fake Job")
        else:
            st.success("✅ Real Job")

    with c2:
        st.metric(
            "Fraud Risk Score",
            f"{risk_score}%"
        )

    with c3:
        if risk_level == "LOW":
            st.success("🟢 LOW")
        elif risk_level == "MEDIUM":
            st.warning("🟡 MEDIUM")
        else:
            st.error("🔴 HIGH")

    # Progress Bar
    st.progress(risk_score / 100)

    st.metric(
        "Model Confidence",
        f"{probability*100:.2f}%"
    )

    # --------------------------------------------------
    # Suspicious Indicators
    # --------------------------------------------------
    st.subheader("⚠️ Suspicious Indicators")

    if indicators:
        for item in indicators:
            st.warning(item)
    else:
        st.success("No suspicious indicators detected.")

    # --------------------------------------------------
    # Fraud Analysis Summary
    # --------------------------------------------------
    st.subheader("📄 Fraud Analysis Summary")

    if prediction == 1:
        st.error(
            f"""
The job posting appears **suspicious**.

**Fraud Risk Score:** {risk_score}%

**Risk Level:** {risk_level}

The system detected **{len(indicators)} suspicious indicator(s)**.

Please verify the company and recruiter before applying.
"""
        )
    else:
        st.success(
            f"""
The job posting appears **legitimate**.

**Fraud Risk Score:** {risk_score}%

**Risk Level:** {risk_level}

No major suspicious indicators were detected.
"""
        )

    # --------------------------------------------------
    # Safety Tips
    # --------------------------------------------------
    st.subheader("🛡️ Safety Tips")

    for tip in get_safety_tips():
        st.write(f"✅ {tip}")

    # --------------------------------------------------
    # Model Performance
    # --------------------------------------------------
    st.divider()

    st.subheader("📈 Model Performance")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Accuracy", "96.67%")
    p2.metric("Precision", "62.30%")
    p3.metric("Recall", "86.74%")
    p4.metric("F1 Score", "72.52%")

st.divider()

st.caption(
    "Developed by Raghav Chauhan | B.Tech IT | NIT Kurukshetra | Big Data Analytics Project"
)