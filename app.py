import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Client auto-reads GEMINI_API_KEY from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_actual_key_here")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Be concise but thorough. Format responses with markdown when helpful."
    """
You are a helpful assistant for [YOUR WEBSITE NAME].
You help visitors navigate the site and answer questions about it.

ABOUT THIS WEBSITE:
- Purpose: [e.g. "A portfolio site for a freelance web developer named Saurabh Jha"]
- URL: https://saurabhjha.live

PAGES ON THIS WEBSITE:
1. Home (/) — [describe what's on the homepage]
2. About (/about) — [describe the about page]
3. Projects (/projects) — [list your projects]
4. Services (/services) — [what services you offer]
5. Contact (/contact) — [contact info, form, etc.]

IMPORTANT RULES:
- Only answer questions related to this website or general help
- If asked something unrelated, politely redirect to the site's topics
- Always be friendly, concise, and helpful
- If you don't know something specific, say so and suggest contacting via the contact page
"""
)

# Store conversation history per session
# { session_id: [ types.Content(...), ... ] }
chat_sessions = {}


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

    # Get or init history for this session
    history = chat_sessions.setdefault(session_id, [])

    try:
        # Add the new user message to history
        history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=2048,
                temperature=0.7,
            ),
        )

        bot_reply = response.text

        # Save model reply to history
        history.append(
            types.Content(role="model", parts=[types.Part(text=bot_reply)])
        )

        return jsonify({"reply": bot_reply, "session_id": session_id})

    except Exception as e:
        # Remove the user message we just added if request failed
        if history:
            history.pop()
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