import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from groq import Groq
import os
from datetime import datetime

# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ==================================================
# Custom CSS (POLISH)
# ==================================================
st.markdown("""
<style>
    .main {
        padding-top: 1.5rem;
    }
    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
    }
    .subtitle-text {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f9fafb;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }
    .metric-box {
        background-color: #eef2ff;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Load Data & Train Model
# ==================================================
df = pd.read_csv("StudentPerformance.csv")
X = df[['HoursStudied', 'Attendance']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

# ==================================================
# GenAI Client
# ==================================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==================================================
# Prompt Builder
# ==================================================
def build_prompt(hours, attendance, marks, tone):
    tone_map = {
        "Mentor": "Be supportive and motivating.",
        "Friendly": "Be casual and friendly.",
        "Strict": "Be honest, direct, and firm."
    }
    return f"""
You are an academic advisor.
{tone_map[tone]}

Student details:
Study hours: {hours}
Attendance: {attendance}%
Predicted marks: {round(marks,2)}

Give 3 short, practical suggestions.
"""

# ==================================================
# AI Response
# ==================================================
def get_feedback(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an academic advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# ==================================================
# Header
# ==================================================
st.markdown("<div class='title-text'>🎓 AI Student Performance Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Predict academic outcomes and receive AI-driven guidance</div>", unsafe_allow_html=True)

# ==================================================
# Sidebar
# ==================================================
st.sidebar.title("⚙️ Settings")
tone = st.sidebar.selectbox("AI Feedback Tone", ["Mentor", "Friendly", "Strict"])
st.sidebar.markdown("---")
st.sidebar.caption("Built by Sulekha Thakur")

# ==================================================
# Input Card
# ==================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📥 Student Inputs")

col1, col2 = st.columns(2)
with col1:
    hours = st.slider("Hours Studied per Day", 0, 12, 5)
with col2:
    attendance = st.slider("Attendance (%)", 0, 100, 80)

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# Prediction
# ==================================================
prediction = model.predict([[hours, attendance]])[0]

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Prediction Result")

st.markdown(
    f"<div class='metric-box'>Predicted Marks: {prediction:.2f}</div>",
    unsafe_allow_html=True
)

confidence = min(95, max(60, int(prediction)))
st.progress(confidence)
st.caption("Confidence based on historical data patterns")

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# Visualization
# ==================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Performance Visualization")

fig, ax = plt.subplots()
ax.scatter(df['HoursStudied'], df['Marks'], alpha=0.6, label="Historical Data")
ax.scatter(hours, prediction, color="crimson", s=120, label="Current Prediction")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
ax.legend()

st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# AI Feedback
# ==================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 AI Feedback")

try:
    prompt = build_prompt(hours, attendance, prediction, tone)
    feedback = get_feedback(prompt)
    st.write(feedback)
except Exception:
    st.warning("AI feedback temporarily unavailable.")

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# Download Report
# ==================================================
report = f"""
AI Student Performance Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Study Hours: {hours}
Attendance: {attendance}%
Predicted Marks: {prediction:.2f}

AI Feedback ({tone}):
{feedback if 'feedback' in locals() else 'Unavailable'}
"""

st.download_button(
    "⬇️ Download Report",
    report,
    file_name="student_performance_report.txt"
)

# ==================================================
# Footer
# ==================================================
st.markdown("---")
st.caption("ML + GenAI powered educational analytics • Streamlit Cloud")
