# 🎓 AI Career Guidance Assistant

A professional, ChatGPT-style **AI Career Guidance Assistant** built using **Streamlit** and **Groq (LLaMA 3)**.  
The application provides personalized career advice, learning roadmaps, resume guidance, and skill-based recommendations through a clean conversational interface.

This project is designed to demonstrate **AI integration, UX thinking, and secure API handling**.

---

## ✨ Features

- 🤖 AI-powered career guidance using **LLaMA 3 (Groq API)**
- 💬 ChatGPT-style conversational UI
- 🆕 New Conversation support
- 🕘 Persistent chat history across sessions
- 🏷️ Auto-generated chat titles
- 👤 User Career Profile (skills, education, interests)
- 🔐 Secure API key handling (environment-based)
- 🎨 Clean, professional, resume-ready UI
- ⚡ Fast responses using Groq inference

---

## 🧠 What the Assistant Can Do

- Suggest suitable career paths based on your skills
- Recommend learning paths and next steps
- Guide internship and job preparation
- Provide resume improvement tips
- Answer career-related questions conversationally

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **LLM:** Groq API (LLaMA 3)
- **State Management:** Streamlit session state
- **Storage:** Local JSON (chat history & profile)
- **Security:** Environment variables (`.env`)

---

## 📂 Project Structure

```
AI_Career_Chatbot/
├── .streamlit/            # Streamlit configuration and theme
├── knowledge_base/        # Documents / data used for AI context
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── utils.py               # Helper / utility functions
```

---

## 🔐 API Key Setup (One-Time)

This app uses the **Groq API**.  
The API key is required **only once** and is never shown in the UI.

### 1️⃣ Get a Groq API Key
- Visit: https://console.groq.com
- Create an API key

### 2️⃣ Create `.env` File
Create a file named `.env` in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

⚠️ **Do NOT commit `.env` to GitHub**

---

## ▶️ How to Run the App

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the App

```bash
streamlit run app.py
```

The app will open in your browser automatically.

---

## 🧪 First-Time Experience

* App loads normally
* If API key is missing → a small setup section appears
* Once configured, the app behaves like **ChatGPT**
* No repeated API key prompts
* Chat works instantly on every launch

---
