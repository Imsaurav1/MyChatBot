import os
import re
import time
import threading
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

# ── Site content catalog (for local suggestion matching, NOT sent to any AI) ─
CATALOG_URL         = os.environ.get("CATALOG_URL", "https://shivmarg.live/abc.json")
CATALOG_TTL_SECONDS = int(os.environ.get("CATALOG_TTL_SECONDS", "1800"))  # 30 min
MAX_SUGGESTIONS     = int(os.environ.get("MAX_SUGGESTIONS", "5"))

# Optional aliases so an English query (e.g. "shiva") also matches catalog
# entries whose title/category fields are Hindi-only. Extend freely.
DEITY_ALIASES = {
    "shiva":     ["शिव", "shankar", "mahadev", "महादेव"],
    "ram":       ["राम", "rama", "ramchandra"],
    "krishna":   ["कृष्ण", "krishn"],
    "hanuman":   ["हनुमान"],
    "durga":     ["दुर्गा"],
    "ganesh":    ["गणेश", "ganpati", "गणपति"],
    "vishnu":    ["विष्णु"],
    "lakshmi":   ["लक्ष्मी", "laxmi"],
    "saraswati": ["सरस्वती"],
    "kali":      ["काली"],
    "surya":     ["सूर्य"],
}

SYSTEM_INSTRUCTION = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Be concise but thorough. Format responses with markdown when helpful."
    "Your Name Is Syneix AI"
)

# In-memory chat history { session_id: [ {role, content} ] }
chat_sessions = {}

# ── Site content catalog cache ────────────────────────────────────────────────
# We fetch your catalog (CATALOG_URL) at most once every CATALOG_TTL_SECONDS and
# keep it in memory. Matching against it is plain Python string scoring — it
# never gets forwarded into an AI prompt, so it costs zero provider tokens no
# matter how often a user chats.
_catalog_cache      = []
_catalog_loaded_at  = 0.0
_catalog_lock       = threading.Lock()

_WORD_RE = re.compile(r"[a-zA-Z\u0900-\u097F]+")  # latin words + devanagari words


def _normalize_catalog(raw):
    """Accept whatever shape the JSON happens to be in: a bare list, a dict
    wrapping a list under a common key, a dict-of-entries, or even a single
    entry object (like the one-item example) — always return a flat list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("items", "data", "results", "pages", "content", "entries"):
            if isinstance(raw.get(key), list):
                return raw[key]
        if raw and all(isinstance(v, dict) for v in raw.values()):
            return list(raw.values())
        if "url" in raw or "title" in raw:
            return [raw]
    return []


def _refresh_catalog():
    global _catalog_cache, _catalog_loaded_at
    try:
        resp = requests.get(CATALOG_URL, timeout=10)
        resp.raise_for_status()
        entries = _normalize_catalog(resp.json())
        if entries:
            _catalog_cache = entries
        print(f"[Catalog] Loaded {len(_catalog_cache)} entries from {CATALOG_URL}")
    except Exception as e:
        print(f"[Catalog] Refresh failed, keeping {len(_catalog_cache)} cached entries: {e}")
    finally:
        _catalog_loaded_at = time.time()


def get_catalog():
    """Return the cached catalog, refreshing it if stale. Pure HTTP GET to
    your own site — not an AI provider call, so this never burns tokens."""
    if not _catalog_cache or (time.time() - _catalog_loaded_at) > CATALOG_TTL_SECONDS:
        with _catalog_lock:
            if not _catalog_cache or (time.time() - _catalog_loaded_at) > CATALOG_TTL_SECONDS:
                _refresh_catalog()
    return _catalog_cache


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _expand_with_aliases(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for tok in tokens:
        expanded.extend(DEITY_ALIASES.get(tok, []))
    return expanded


def _score_entry(tokens: list[str], entry: dict) -> int:
    keywords = [str(k).lower() for k in entry.get("keywords", []) or []]
    hashtags = [str(h).lower() for h in entry.get("hashtags", []) or []]
    blob = " ".join(
        str(entry.get(f, "") or "")
        for f in ("title", "titleEng", "name", "eng", "typeLabel", "category")
    ).lower()
    preview = str(entry.get("preview", "") or "").lower()

    score = 0
    for tok in tokens:
        if tok in keywords:
            score += 5
        elif any(tok in k for k in keywords):
            score += 3
        if tok in hashtags:
            score += 4
        elif any(tok in h for h in hashtags):
            score += 2
        if tok in blob:
            score += 3
        if tok in preview:
            score += 1
    if score > 0 and entry.get("featured"):
        score += 1  # small tiebreaker nudge for featured content, only on real matches
    return score


def get_suggestions(query: str, limit: int = None) -> list[dict]:
    """Local-only matching against the cached catalog — no AI call happens
    here, so suggestions are effectively free regardless of chat volume."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    tokens = _expand_with_aliases(tokens)

    scored = []
    for entry in get_catalog():
        s = _score_entry(tokens, entry)
        if s > 0:
            scored.append((s, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {
            "title":    e.get("title") or e.get("name"),
            "titleEng": e.get("titleEng") or e.get("eng"),
            "url":      e.get("url"),
            "symbol":   e.get("symbol"),
            "img":      e.get("img") or e.get("imageUrl"),
            "category": e.get("category"),
        }
        for _, e in scored[: (limit or MAX_SUGGESTIONS)]
    ]


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

    # Check the site catalog FIRST — plain Python string matching, no AI
    # provider is touched for this, so it costs zero tokens either way.
    suggestions = get_suggestions(user_message)

    history = chat_sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})

    reply, provider = get_ai_response(history)

    history.append({"role": "assistant", "content": reply})

    return jsonify({
        "reply": reply,
        "provider": provider,
        "session_id": session_id,
        "suggestions": suggestions,
    })


@app.route("/reset", methods=["POST"])
def reset():
    data       = request.json or {}
    session_id = data.get("session_id", "default")
    chat_sessions.pop(session_id, None)
    return jsonify({"status": "reset", "session_id": session_id})


@app.route("/catalog/refresh", methods=["POST"])
def refresh_catalog_route():
    _refresh_catalog()
    return jsonify({"status": "refreshed", "count": len(_catalog_cache), "source": CATALOG_URL})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "providers": [n for n, _ in PROVIDERS]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)