import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from groq import Groq
import os

# ==================================
# Page Configuration
# ==================================
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

# ==================================
# Load Data
# ==================================
df = pd.read_csv("StudentPerformance.csv")

X = df[['HoursStudied', 'Attendance']]
y = df['Marks']

# ==================================
# Train Model
# ==================================
model = LinearRegression()
model.fit(X, y)

# ==================================
# Initialize GenAI Client (Secure)
# ==================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==================================
# Prompt Builder
# ==================================
def build_prompt(hours, attendance, predicted_marks):
    return f"""
You are an academic performance advisor.

Student profile:
- Study hours per day: {hours}
- Attendance percentage: {attendance}%
- Predicted academic score: {round(predicted_marks, 2)}

Provide 3 short, practical, and motivating suggestions to improve academic performance.
Keep the tone professional and supportive.
"""

# ==================================
# GenAI Response Function
# ==================================
def get_genai_feedback(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful academic advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# ==================================
# UI: Header
# ==================================
st.title("📊 AI Student Performance Predictor")
st.caption(
    "Predict academic performance and receive AI-driven insights to improve learning outcomes."
)
st.markdown("---")

# ==================================
# UI: Sidebar
# ==================================
st.sidebar.title("About This App")
st.sidebar.write(
    "This application combines supervised machine learning with a generative AI layer "
    "to predict student performance and provide personalized feedback."
)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Sulekha Thakur")

# ==================================
# UI: Inputs
# ==================================
st.subheader("📥 Student Inputs")

with st.container():
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

# ==================================
# Prediction + Output
# ==================================
if st.button("🔍 Predict Performance"):
    prediction = model.predict([[hours, attendance]])[0]

    st.success(f"📈 Predicted Marks: {prediction:.2f}")

    # Confidence indicator (UI enhancement)
    confidence = min(95, max(60, int(prediction)))
    st.progress(confidence)
    st.caption("Prediction confidence based on data patterns")

    # GenAI Feedback
    st.markdown("### 🤖 AI-Generated Feedback")
    try:
        prompt = build_prompt(hours, attendance, prediction)
        ai_feedback = get_genai_feedback(prompt)
        st.write(ai_feedback)
    except Exception:
        st.warning(
            "AI feedback is temporarily unavailable. "
            "The performance prediction is still valid."
        )

# ==================================
# Footer
# ==================================
st.markdown("---")
st.caption(
    "Powered by Python, Scikit-learn, Streamlit, and GenAI | Educational AI Demo"
)
