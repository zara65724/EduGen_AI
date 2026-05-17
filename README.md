# 🎓 EduGen_AI – Smart Study Assistant

<div align="center">

![EduGen_AI](https://img.shields.io/badge/EduGen__AI-Study%20Assistant-6C63FF?style=for-the-badge&logo=bookstack&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-F55036?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3144?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered RAG study assistant — upload PDFs and chat, summarize, generate MCQs, and extract key insights instantly.**

</div>

---

## 📌 Overview

**EduGen_AI** is a full-stack AI web application using Python, Streamlit, LangChain, **Groq API** (LLaMA 3.3 70B), and FAISS vector database.

It uses **Retrieval-Augmented Generation (RAG)** — every answer is grounded in your actual document, not hallucinated.

**Groq is 100% free** — no credit card required. It's also much faster than most other LLM APIs.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Upload** | Upload one or multiple PDFs simultaneously |
| 💬 **Conversational Q&A** | Ask questions with full chat history |
| 📋 **Smart Summarization** | Brief / Standard / Detailed modes |
| 📝 **MCQ Generator** | Auto-generate quizzes with difficulty control |
| 🔍 **Key Point Extraction** | Extract top concepts instantly |
| ⬇️ **Download Results** | Export summaries, MCQs, key points as `.txt` |
| 🎨 **Premium Dark UI** | Glassmorphism design, smooth animations |
| ☁️ **Deployment Ready** | Streamlit Cloud compatible |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit + Custom CSS |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| RAG Orchestration | LangChain `ConversationalRetrievalChain` |
| Vector Store | FAISS (in-memory) |
| PDF Parsing | PyPDF |

---

## ⚡ Quick Start (Local)

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/EduGen_AI.git
cd EduGen_AI
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90MB). This only happens once.

### Step 4 — Add your Groq API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

App opens at `http://localhost:8501` 🎉

---

## 🔑 Get Your FREE Groq API Key

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up with Google or email (free, no credit card)
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_`)
5. Paste it into:
   - `.env` file for local use
   - Streamlit Cloud Secrets for deployment

---

## 📤 How to Upload to GitHub

Follow these steps **exactly** — step by step:

### Step 1 — Install Git

Download from [git-scm.com](https://git-scm.com/downloads) and install.

Verify:
```bash
git --version
```

### Step 2 — Create GitHub Account

Go to [github.com](https://github.com) → Sign Up (free).

### Step 3 — Create a New Repository on GitHub

1. Click the **"+"** icon (top right) → **"New repository"**
2. Repository name: `EduGen_AI`
3. Set to **Public**
4. Do NOT check "Add README" (you already have one)
5. Click **"Create repository"**

### Step 4 — Create .gitignore (important!)

Make sure your `.gitignore` file contains:

```
.env
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

This prevents your API key from being uploaded.

### Step 5 — Initialize Git in your project folder

Open terminal/cmd in your project folder:

```bash
git init
git add .
git commit -m "Initial commit - EduGen_AI"
```

### Step 6 — Connect to GitHub and Push

Copy the commands from your GitHub repo page, or use these (replace YOUR_USERNAME):

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/EduGen_AI.git
git push -u origin main
```

Enter your GitHub username and password (or Personal Access Token).

### Step 7 — Verify

Go to `https://github.com/YOUR_USERNAME/EduGen_AI` — you should see all your files! ✅

---

## ☁️ Deploy on Streamlit Cloud (Free)

### Step 1 — Go to Streamlit Cloud

Visit [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub.

### Step 2 — Create New App

1. Click **"New app"**
2. Select your `EduGen_AI` repository
3. Branch: `main`
4. Main file path: `app.py`

### Step 3 — Add Your API Key as Secret

1. Click **"Advanced settings"**
2. Under **Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

3. Click **"Deploy!"**

Your app will be live at:
`https://your-app-name.streamlit.app` 🚀

---

## 📂 Project Structure

```
EduGen_AI/
│
├── app.py                    # Main Streamlit application
├── utils.py                  # Backend: PDF, RAG, Groq AI logic
├── style.css                 # Premium glassmorphism CSS
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── .gitignore                # Protects secrets from GitHub
├── README.md                 # This file
│
└── .streamlit/
    ├── config.toml           # Dark theme config
    └── secrets.toml.example  # Cloud secrets template
```

---

## 🧑‍💻 How RAG Works in This App

```
User uploads PDF(s)
        ↓
Text extracted with PyPDF
        ↓
Text split into overlapping chunks (LangChain)
        ↓
Chunks embedded locally (HuggingFace MiniLM — FREE)
        ↓
Vectors stored in FAISS (in-memory)
        ↓
User asks a question
        ↓
Query embedded → Top-5 similar chunks retrieved
        ↓
Chunks + question sent to Groq (LLaMA 3.3 70B)
        ↓
Grounded, accurate answer returned ✅
```

---

## 🐛 Troubleshooting

**"Invalid API key"**
→ Check [console.groq.com/keys](https://console.groq.com/keys). Key must start with `gsk_`.

**Slow first run**
→ Embedding model (~90MB) downloads once on first use. Subsequent runs are fast.

**"Could not extract text"**
→ Your PDF may be a scanned image. Use a text-based PDF.

**Streamlit Cloud error: Module not found**
→ Make sure `requirements.txt` has all packages listed correctly.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Made with ❤️ | EduGen_AI — Learn Smarter, Not Harder
</div>
