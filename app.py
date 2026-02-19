import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY",     "")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY",       "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

SYSTEM_INSTRUCTION = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Be concise but thorough. Format responses with markdown when helpful."
)

# In-memory chat history { session_id: [ {role, content} ] }
chat_sessions = {}

# ── Gemini SDK client (initialized once at startup) ───────────────────────────
gemini_client = None
gemini_types  = None

if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types as _gt
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_types  = _gt
        print("[Gemini] SDK client ready ✓")
    except Exception as e:
        print(f"[Gemini] SDK init failed: {e}")


# ── Provider: Gemini 1.5 Flash (google-genai SDK) ────────────────────────────
def try_gemini(messages: list) -> str | None:
    if not gemini_client:
        return None
    try:
        contents = []
        for m in messages:
            sdk_role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                gemini_types.Content(
                    role=sdk_role,
                    parts=[gemini_types.Part(text=m["content"])]
                )
            )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=gemini_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=2048,
                temperature=0.7,
            ),
        )
        return response.text
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            print("[Gemini] Quota exceeded → trying next provider...")
        else:
            print(f"[Gemini] Error: {e}")
        return None


# ── Provider: Groq — Llama 3.3 70B ───────────────────────────────────────────
def try_groq(messages: list) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        full_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}] + messages
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": full_messages,
                "max_tokens": 2048,
                "temperature": 0.7,
            },
            timeout=20,
        )
        if resp.status_code == 429:
            print("[Groq] Rate limited → trying next provider...")
            return None
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Groq] Error: {e}")
        return None


# ── Provider: OpenRouter — free Llama model ───────────────────────────────────
def try_openrouter(messages: list) -> str | None:
    if not OPENROUTER_API_KEY:
        return None
    try:
        full_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}] + messages
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mychatbot-7v19.onrender.com",
            },
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": full_messages,
                "max_tokens": 2048,
            },
            timeout=30,
        )
        if resp.status_code == 429:
            print("[OpenRouter] Rate limited → trying next provider...")
            return None
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[OpenRouter] Error: {e}")
        return None


# ── Provider: Cloudflare Workers AI ──────────────────────────────────────────
def try_cloudflare(messages: list) -> str | None:
    cf_account = os.environ.get("CF_ACCOUNT_ID", "")
    cf_token   = os.environ.get("CF_API_TOKEN",  "")
    if not cf_account or not cf_token:
        return None
    try:
        resp = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/@cf/meta/llama-3.1-8b-instruct",
            headers={"Authorization": f"Bearer {cf_token}"},
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    *messages,
                ]
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["result"]["response"]
    except Exception as e:
        print(f"[Cloudflare] Error: {e}")
        return None


# ── Fallback chain ────────────────────────────────────────────────────────────
PROVIDERS = [
    ("Gemini 1.5 Flash", try_gemini),
    ("Groq Llama 3.3",   try_groq),
    ("OpenRouter Llama", try_openrouter),
    ("Cloudflare AI",    try_cloudflare),
]

def get_ai_response(messages: list) -> tuple[str, str]:
    for name, fn in PROVIDERS:
        result = fn(messages)
        if result:
            return result, name
    return (
        "⚠️ All AI providers are currently rate limited. Please try again in a minute.",
        "none",
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data         = request.json or {}
    user_message = data.get("message", "").strip()
    session_id   = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    history = chat_sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})

    reply, provider = get_ai_response(history)

    history.append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply, "provider": provider, "session_id": session_id})


@app.route("/reset", methods=["POST"])
def reset():
    data       = request.json or {}
    session_id = data.get("session_id", "default")
    chat_sessions.pop(session_id, None)
    return jsonify({"status": "reset", "session_id": session_id})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "providers": [n for n, _ in PROVIDERS]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)