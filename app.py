import os
import base64
import sqlite3
import requests
import io
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
from pydub import AudioSegment
import imageio_ffmpeg
_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = _ffmpeg_exe
AudioSegment.ffmpeg    = _ffmpeg_exe
AudioSegment.ffprobe   = _ffmpeg_exe

app = Flask(__name__)
CORS(app)

# Rasa Endpoints - using environment variables for Railway deployment
RASA_HOST      = os.getenv("RASA_URL", "http://localhost:5005")
RASA_URL       = f"{RASA_HOST}/webhooks/rest/webhook"
RASA_PARSE_URL = f"{RASA_HOST}/model/parse"

# /tmp is always writable on Railway (and any container runtime)
DB_PATH = os.getenv("DB_PATH", "/tmp/unknown_questions.db")

# Confidence threshold — questions below this are logged as fallback
FALLBACK_THRESHOLD = 0.10

FALLBACK_PHRASES = [
    "i didn't quite understand",
    "i don't understand",
    "i'm not sure what you mean",
    "i didn't get that",
    "could you rephrase",
    "i cannot help with that",
    "your question has been recorded",
]

# Initialize tools
recognizer = sr.Recognizer()
translator = Translator()


# ─────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────

def init_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS unknown_questions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            question      TEXT    NOT NULL,
            language      TEXT    DEFAULT 'en',
            translated_en TEXT,
            detected_intent TEXT,
            confidence    REAL,
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
            reviewed      INTEGER DEFAULT 0,
            added_to_training INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Database ready at: {DB_PATH}")


def log_unknown_question(question, language, translated_en, detected_intent, confidence):
    """Save an unrecognized question to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO unknown_questions
                (question, language, translated_en, detected_intent, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (question, language, translated_en, detected_intent, confidence, datetime.now()))
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        print(f"[DB] Logged fallback question (id={new_id}): {question}")
        return new_id
    except Exception as e:
        print(f"[DB ERROR] Could not log question: {e}")
        return None


# ─────────────────────────────────────────────
#  RASA HELPERS
# ─────────────────────────────────────────────

def query_rasa(text):
    """
    Send message to Rasa and determine if it's a fallback.
    Returns (bot_reply, intent_name, confidence, is_fallback).
    """
    bot_reply   = "Sorry, I didn't understand that."
    intent_name = "unknown"
    confidence  = 0.0
    is_fallback = False

    # ── Step 1: get intent confidence from /model/parse ──
    try:
        r = requests.post(RASA_PARSE_URL, json={"text": text}, timeout=10)
        parsed      = r.json()
        top_intent  = parsed.get("intent", {})
        intent_name = top_intent.get("name", "unknown")
        confidence  = top_intent.get("confidence", 0.0)

        if intent_name in ("nlu_fallback", "out_of_scope"):
            is_fallback = True
            print(f"[RASA] Fallback intent detected: {intent_name}")
        elif confidence < FALLBACK_THRESHOLD:
            is_fallback = True
            print(f"[RASA] Low confidence ({confidence:.2f}) → fallback")

    except Exception as e:
        print(f"[RASA PARSE WARNING] /model/parse unreachable: {e}")

    # ── Step 2: get bot reply from webhook ──
    try:
        r2   = requests.post(RASA_URL, json={"sender": "user", "message": text}, timeout=10)
        data = r2.json()
        if data:
            bot_reply = data[0].get("text", bot_reply)
        else:
            is_fallback = True
            print("[RASA] Empty webhook response → fallback")
    except Exception as e:
        print(f"[RASA ERROR] {e}")
        is_fallback = True
        bot_reply = "Rasa server is down."

    # ── Step 3: reply text safety net ──
    if any(phrase in bot_reply.lower() for phrase in FALLBACK_PHRASES):
        is_fallback = True
        print(f"[RASA] Fallback phrase detected in reply")

    print(f"[RASA] intent='{intent_name}' conf={confidence:.2f} fallback={is_fallback} reply='{bot_reply[:80]}'")
    return bot_reply, intent_name, confidence, is_fallback


# ─────────────────────────────────────────────
#  MAIN AUDIO ROUTE
# ─────────────────────────────────────────────

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    lang_code = request.form.get('language', 'en')

    # 1. Convert audio to WAV
    try:
        audio = AudioSegment.from_file(audio_file)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500

    # 2. Speech to Text
    user_text = ""
    rec_lang = 'ur-PK' if lang_code == 'ur' else 'en-US'
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        try:
            user_text = recognizer.recognize_google(audio_data, language=rec_lang)
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 400
        except sr.RequestError:
            return jsonify({"error": "Speech service unavailable"}), 503

    print(f"[STT] User said ({lang_code}): {user_text}")

    # 3. Translate Urdu → English for Rasa
    rasa_input = user_text
    translated_en = None
    if lang_code == 'ur':
        translated_en = translator.translate(user_text, src='ur', dest='en').text
        rasa_input = translated_en
        print(f"[TRANSLATE] Urdu → English: {rasa_input}")

    # 4. Single Rasa call
    bot_response_en, detected_intent, confidence, is_fallback = query_rasa(rasa_input)
    print(f"[RASA] Response: {bot_response_en}")

    # 5. Log to DB and override response if fallback
    if is_fallback:
        log_unknown_question(
            question=user_text,
            language=lang_code,
            translated_en=translated_en if lang_code == 'ur' else user_text,
            detected_intent=detected_intent,
            confidence=confidence
        )
        if lang_code == 'ur':
            bot_response_en = "معذرت، میں آپ کا سوال نہیں سمجھ سکا۔ براہ کرم دوسرا سوال پوچھیں۔"
        else:
            bot_response_en = "I'm sorry, I don't have an answer for that. Please try asking another question."

    # 6. Translate response back to Urdu if needed
    final_response = bot_response_en
    if lang_code == 'ur' and not is_fallback:
        final_response = translator.translate(bot_response_en, src='en', dest='ur').text
        print(f"[TRANSLATE] English → Urdu: {final_response}")

    # 7. Text to Speech
    tts_lang = 'ur' if lang_code == 'ur' else 'en'
    tts = gTTS(text=final_response, lang=tts_lang, slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')

    return jsonify({
        "user_text": user_text,
        "bot_text": final_response,
        "audio_base64": audio_base64,
        "is_fallback": is_fallback,
        "intent": detected_intent,
        "confidence": round(confidence, 3)
    })


# ─────────────────────────────────────────────
#  ADMIN API ROUTES
# ─────────────────────────────────────────────

@app.route('/admin/questions', methods=['GET'])
def get_questions():
    reviewed_filter = request.args.get('reviewed')
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if reviewed_filter is not None:
        cursor.execute(
            "SELECT * FROM unknown_questions WHERE reviewed=? ORDER BY timestamp DESC",
            (int(reviewed_filter),)
        )
    else:
        cursor.execute("SELECT * FROM unknown_questions ORDER BY timestamp DESC")

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route('/admin/questions/stats', methods=['GET'])
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM unknown_questions")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM unknown_questions WHERE reviewed=0")
    pending = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM unknown_questions WHERE added_to_training=1")
    trained = cursor.fetchone()[0]
    conn.close()
    return jsonify({"total": total, "pending": pending, "added_to_training": trained})


@app.route('/admin/questions/<int:qid>/review', methods=['PUT'])
def mark_reviewed(qid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE unknown_questions SET reviewed=1 WHERE id=?", (qid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "id": qid})


@app.route('/admin/questions/<int:qid>/trained', methods=['PUT'])
def mark_trained(qid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE unknown_questions SET added_to_training=1, reviewed=1 WHERE id=?", (qid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "id": qid})


@app.route('/admin/questions/<int:qid>', methods=['DELETE'])
def delete_question(qid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM unknown_questions WHERE id=?", (qid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted", "id": qid})


@app.route('/admin/questions/export', methods=['GET'])
def export_questions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT translated_en FROM unknown_questions WHERE reviewed=0 ORDER BY timestamp DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    lines = ["# Unreviewed Fallback Questions — Add intents & examples to nlu.yml\n"]
    for i, row in enumerate(rows, 1):
        lines.append(f"{i}. {row['translated_en']}")

    text_content = "\n".join(lines)
    buf = io.BytesIO(text_content.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain',
                     as_attachment=True, download_name='fallback_questions.txt')


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

# Called at module level so Gunicorn triggers it on import
init_db()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=False, port=port)
