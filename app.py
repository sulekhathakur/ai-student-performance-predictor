import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("StudentPerformance.csv")

X = df[['HoursStudied', 'Attendance']]
y = df['Marks']

# ===============================
# Train ML Model
# ===============================
model = LinearRegression()
model.fit(X, y)

# ===============================
# GenAI-style Prompt Builder
# ===============================
def build_prompt(hours, attendance, predicted_marks):
    return f"""
You are an academic performance advisor.

Student details:
- Study hours per day: {hours}
- Attendance: {attendance}%
- Predicted marks: {round(predicted_marks, 2)}

Give 3 short, practical suggestions to improve performance.
"""

# ===============================
# Simulated GenAI Response
# ===============================
def simulated_llm_response(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful academic advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("📊 Student Performance Predictor")
st.write("An AI-powered app combining Machine Learning and GenAI-style feedback.")

hours = st.number_input(
    "Hours Studied per Day",
    min_value=0,
    max_value=12,
    value=5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

# ===============================
# Prediction + GenAI Feedback
# ===============================
if st.button("Predict Performance"):
    prediction = model.predict([[hours, attendance]])[0]

    st.success(f"🎯 Predicted Marks: {prediction:.2f}")

    prompt = build_prompt(hours, attendance, prediction)
    ai_feedback = simulated_llm_response(prompt)

    st.subheader("🤖 AI-Generated Feedback")
    st.text(ai_feedback)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built using Python, Scikit-learn, Streamlit | ML + GenAI Concepts")
