# AI Student Performance Predictor

An AI-powered web application that predicts student academic performance and provides personalized, actionable feedback to support learning improvement.

🔗 **Live Demo:**
👉 [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)
*(Note: The app may take a few seconds to load if inactive.)*

---

## Overview

This project combines supervised machine learning with a generative AI layer to deliver both **data-driven predictions** and **human-like academic guidance**. Users input basic study metrics, and the system predicts expected performance while offering tailored suggestions to help improve outcomes.

The focus of this project is practical, real-world application of AI concepts, including deployment, UI design, and secure integration of large language models.

---

## Key Features

* Predicts student marks based on study hours and attendance
* Uses supervised machine learning (Linear Regression)
* Generates personalized academic feedback using GenAI
* Clean, interactive web interface built with Streamlit
* Configurable AI tone (mentor, friendly, or strict)
* Visual performance insights and downloadable reports
* Secure handling of API keys (no secrets in codebase)

---

## Technology Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Streamlit**
* **Groq LLM API**

---

## Project Structure

```
ai-student-performance-predictor/
├── app.py
├── StudentPerformance.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## How the Application Works

1. User provides study hours and attendance percentage
2. A machine learning model predicts expected academic performance
3. A structured prompt is generated based on user input
4. The GenAI layer produces concise, personalized improvement suggestions
5. Users can visualize performance trends and download a report

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/sulekhathakur/ai-student-performance-predictor.git
cd ai-student-performance-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the GenAI API key

**Windows (PowerShell):**

```powershell
setx GROQ_API_KEY "your_api_key_here"
```

Restart the terminal after setting the variable.

### 4. Run the application

```bash
python -m streamlit run app.py
```

---

## Deployment

The application is deployed using **Streamlit Cloud**.
Dependencies are installed automatically via `requirements.txt`, and sensitive credentials are managed securely using environment variables or platform secrets.

---

## Design & Engineering Notes

* The GenAI component is modular and provider-agnostic
* Core ML prediction works independently of the GenAI layer
* The UI emphasizes clarity, usability, and real-world product thinking
* Designed to be easily extendable with additional features or models

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Sulekha Thakur**
Final Year B.Sc. Computer Science
Interests: Applied AI, Generative AI, AI-powered product development

---



