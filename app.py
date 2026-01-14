import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    p {
        color: #cfcfcf;
    }
    .card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .result {
        background-color: #0f5132;
        padding: 15px;
        border-radius: 10px;
        color: #d1e7dd;
        font-size: 18px;
    }
    .ai-box {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4dabf7;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📊 AI Student Performance Predictor")
st.write(
    "Predict academic performance and receive **AI-driven insights** "
    "to improve learning outcomes."
)

st.divider()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("StudentPerformance.csv")

X = df[["Hours_Studied", "Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

# ---------------- INPUT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧾 Student Inputs")

col1, col2 = st.columns(2)

with col1:
    hours = st.number_input(
        "Hours Studied per Day",
        min_value=0,
        max_value=12,
        value=5
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80
    )

predict_btn = st.button("🔍 Predict Performance")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:
    prediction = model.predict([[hours, attendance]])[0]

    st.markdown(
        f"<div class='result'>✅ <b>Predicted Marks:</b> {prediction:.2f}</div>",
        unsafe_allow_html=True
    )

    # ---------------- AI FEEDBACK (SAFE MODE) ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 AI Academic Insight")

    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""
        A student studies {hours} hours per day and has {attendance}% attendance.
        Their predicted marks are {prediction:.2f}.
        Give short, practical, motivating feedback to improve performance.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        ai_text = response.choices[0].message.content

    except Exception:
        ai_text = (
            "Focus on maintaining consistent study hours, improving class engagement, "
            "and revising weak topics regularly to boost academic performance."
        )

    st.markdown(
        f"<div class='ai-box'>{ai_text}</div>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.caption("Built with ❤️ using Python, Machine Learning & Generative AI")
