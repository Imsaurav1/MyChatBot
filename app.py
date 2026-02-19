import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent?key=" + (GEMINI_API_KEY or "")
)

SYSTEM_INSTRUCTION = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Be concise but thorough. Format responses with markdown when helpful."
)

# Store conversation histories per session: { session_id: [ {role, parts} ] }
chat_sessions = {}


def build_payload(history: list, user_message: str) -> dict:
    """Build the Gemini REST API payload with full history."""
    contents = []

    # Prepend system instruction as first user/model exchange if history is empty
    if not history:
        contents.append({"role": "user",  "parts": [{"text": SYSTEM_INSTRUCTION}]})
        contents.append({"role": "model", "parts": [{"text": "Understood! I'm ready to help."}]})

    contents.extend(history)
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return {"contents": contents}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set on server."}), 500

    history = chat_sessions.setdefault(session_id, [])

    try:
        payload = build_payload(history, user_message)
        resp = requests.post(GEMINI_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        bot_reply = (
            result["candidates"][0]["content"]["parts"][0]["text"]
        )

        # Persist history
        history.append({"role": "user",  "parts": [{"text": user_message}]})
        history.append({"role": "model", "parts": [{"text": bot_reply}]})

        return jsonify({"reply": bot_reply, "session_id": session_id})

    except requests.exceptions.HTTPError as e:
        err_body = e.response.text if e.response else str(e)
        return jsonify({"error": f"Gemini API error: {err_body}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    data = request.json
    session_id = data.get("session_id", "default")
    chat_sessions.pop(session_id, None)
    return jsonify({"status": "reset", "session_id": session_id})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)