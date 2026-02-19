import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ================== APP SETUP ==================

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GEMINI_BASE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

# ================== SYSTEM INSTRUCTION ==================

SYSTEM_INSTRUCTION = """
You are a helpful, friendly, and knowledgeable AI assistant.
Be concise but thorough. Format responses with markdown when helpful.

You are a helpful assistant for [YOUR WEBSITE NAME].
You help visitors navigate the site and answer questions about it.

ABOUT THIS WEBSITE:
- Purpose: A portfolio website
- URL: https://yourwebsite.com

PAGES:
1. Home (/)
2. About (/about)
3. Projects (/projects)
4. Services (/services)
5. Contact (/contact)

IMPORTANT RULES:
- Only answer questions related to this website or general help.
- If asked something unrelated, politely redirect to website topics.
- Always be friendly and concise.
- If unsure, suggest contacting via the contact page.
"""

# ================== SESSION STORAGE ==================

# WARNING: In production, use Redis or DB instead of dictionary
chat_sessions = {}

# ================== HELPER FUNCTION ==================

def build_payload(history: list, user_message: str) -> dict:
    contents = []

    # Inject system instruction at first conversation
    if not history:
        contents.append({
            "role": "user",
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Understood! I'm ready to help."}]
        })

    contents.extend(history)

    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    return {"contents": contents}

# ================== ROUTES ==================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set on server."}), 500

    data = request.json
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    history = chat_sessions.setdefault(session_id, [])

    try:
        payload = build_payload(history, user_message)

        url = f"{GEMINI_BASE}?key={GEMINI_API_KEY}"

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()

        bot_reply = result["candidates"][0]["content"]["parts"][0]["text"]

        # Save conversation history
        history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        history.append({
            "role": "model",
            "parts": [{"text": bot_reply}]
        })

        return jsonify({
            "reply": bot_reply,
            "session_id": session_id
        })

    except requests.exceptions.HTTPError as e:
        err_body = e.response.text if e.response else str(e)
        return jsonify({"error": f"Gemini API error: {err_body}"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    data = request.json or {}
    session_id = data.get("session_id", "default")

    chat_sessions.pop(session_id, None)

    return jsonify({
        "status": "reset",
        "session_id": session_id
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ================== MAIN ==================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
