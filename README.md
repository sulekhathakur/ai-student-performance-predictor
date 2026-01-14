
# AI Student Performance Predictor (ML + GenAI)

An AI-powered web application that predicts student performance using supervised machine learning and provides personalized improvement suggestions using a GenAI layer.

## 🚀 Features

* Predicts student marks based on **hours studied** and **attendance**
* Built using **Linear Regression** (supervised ML)
* Interactive **Streamlit** web interface
* Integrated **GenAI (LLM)** to generate personalized academic feedback
* Secure API key handling using environment variables / secrets

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Streamlit**
* **Groq LLM API**

---

## 📂 Project Structure

```
ai-student-performance-predictor/
│
├── app.py                  # Streamlit application
├── StudentPerformance.csv  # Dataset
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
└── .gitignore              # Ignored files & secrets
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sulekhathakur/ai-student-performance-predictor.git
cd ai-student-performance-predictor
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set environment variable (Groq API key)

**Windows (PowerShell):**

```powershell
setx GROQ_API_KEY "your_api_key_here"
```

Restart terminal after setting.

### 4️⃣ Run the app

```bash
python -m streamlit run app.py
```

---

## 🤖 How the AI Works

1. User inputs study hours and attendance
2. ML model predicts expected marks
3. A prompt is dynamically generated
4. GenAI (LLM) provides actionable improvement suggestions

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud** and automatically installs dependencies from `requirements.txt`.

---

## 📌 Notes

* GenAI layer is **API-based and provider-agnostic**
* If the LLM API is unavailable, the ML prediction still functions
* No API keys are stored in the repository

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙋‍♀️ Author

**Sulekha Thakur**
B.Sc. Computer Science (Final Year)
Aspiring AI / GenAI Engineer



