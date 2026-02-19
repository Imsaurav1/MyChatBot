# 🤖 AI Chatbot — Powered by Gemini 1.5 Flash (Free!)

A full-stack AI chatbot built with Python/Flask + Google Gemini API.  
Beautiful dark-themed UI with markdown rendering, code highlighting, and chat history.

---

## 📁 Project Structure

```
chatbot/
├── app.py              ← Flask backend
├── requirements.txt    ← Python dependencies
├── Procfile            ← Render start command
├── render.yaml         ← Render deployment config
└── templates/
    └── index.html      ← Full chat UI
```

---

## 🚀 Step-by-Step Deploy Guide

### Step 1: Get a FREE Gemini API Key
1. Go to → https://aistudio.google.com/app/apikey
2. Click **"Create API Key"**
3. Copy the key (looks like: `AIzaSy...`)
4. **Free tier**: 15 requests/min, 1M tokens/day — plenty!

### Step 2: Push Code to GitHub
```bash
git init
git add .
git commit -m "Initial chatbot"
git branch -M main
# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/chatbot.git
git push -u origin main
```

### Step 3: Deploy to Render
1. Go to → https://render.com and sign up (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account → select your repo
4. Render auto-detects settings from `render.yaml`, but verify:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free
5. Under **"Environment Variables"**, add:
   - Key: `GEMINI_API_KEY`
   - Value: `AIzaSy...` (your key from Step 1)
6. Click **"Deploy Web Service"**
7. Wait ~2 minutes → your chatbot is LIVE at `https://your-app.onrender.com`!

---

## 🧪 Run Locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"   # Mac/Linux
# set GEMINI_API_KEY=your_key_here       # Windows
python app.py
```
Open → http://localhost:5000

---

## ✨ Features
- 💬 Multi-turn conversation memory per session
- 🎨 Beautiful dark UI with gradient accents
- 📝 Full Markdown rendering (tables, code, lists)
- 🔦 Syntax highlighting for code blocks
- 💡 Suggested prompts on welcome screen
- ↺ New Chat / reset button
- 📱 Mobile responsive

## 🔧 Customize
- Change personality → edit `system_instruction` in `app.py`
- Change model → replace `gemini-1.5-flash` with `gemini-1.5-pro` (more powerful, still free)
- Change theme colors → edit CSS variables at top of `index.html`
