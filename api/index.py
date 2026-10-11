import os
import base64
import tempfile
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
import requests

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Fetch API keys from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MURF_API_KEY = os.environ.get("MURF_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")
if not MURF_API_KEY:
    raise ValueError("MURF_API_KEY is not set in environment variables.")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

PROMPTS = {
    "Summary": """
You are a professional tourist guide.
Provide a high-level overview of "{place}" in {language}.

Focus on:
- The historical significance
- Why the place is famous
- Key architectural or cultural highlights

Keep the explanation concise, engaging, and easy to follow.
Avoid excessive details and dates.
Limit the response to around 200 words.

Respond ONLY in {language}.
""",
    "Detailed": """
You are a professional tourist guide.
Provide a detailed and immersive explanation of "{place}" in {language}.

Cover:
- Historical background and timeline
- Architectural design and unique features
- Cultural importance and notable events
- Interesting facts and visitor insights

Explain concepts clearly and in a storytelling manner.
Include relevant details and examples to create a rich experience.
Limit the response to around 400 words.

Respond ONLY in {language}.
"""
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Server running", "endpoints": ["/generate-audio-guide"]}), 200

def generate_speech(text, voice_id, locale):
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "voice_id": voice_id,
        "text": text,
        "locale": locale,
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO"
    }

    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Murf API Error: {response.status_code} - {response.text}")

    # Write stream to a temporary audio file
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        with open(temp_audio.name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return temp_audio.name
    except Exception:
        if os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)
        raise

def generate_description(place, answer_type, language):
    if answer_type not in PROMPTS:
        answer_type = "Summary"

    prompt = PROMPTS[answer_type].format(place=place, language=language)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

@app.route("/generate-audio-guide", methods=["POST"])
def generate_audio_guide():
    data = request.get_json(silent=True) or {}

    required_fields = ["place", "answerType", "language", "voiceId", "locale"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    place = data["place"]
    answer_type = data["answerType"]
    language = data["language"]
    voice_id = data["voiceId"]
    locale = data["locale"]

    audio_path = None
    try:
        # 1. Generate textual description
        text_description = generate_description(place, answer_type, language)

        # 2. Convert text to speech
        audio_path = generate_speech(text_description, voice_id, locale)

        # 3. Read and encode audio to base64
        with open(audio_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        return jsonify({
            "description": text_description,
            "audioBase64": encoded_audio
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup temporary audio file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)