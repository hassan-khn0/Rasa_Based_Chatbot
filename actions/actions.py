import os
import sqlite3
from datetime import datetime
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# ── DB path: same folder as this actions.py ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "unknown_questions.db")

FALLBACK_THRESHOLD = 0.70


def ensure_db():
    """Create the DB + table if missing."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS unknown_questions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            question           TEXT    NOT NULL,
            language           TEXT    DEFAULT 'en',
            translated_en      TEXT,
            detected_intent    TEXT,
            confidence         REAL,
            timestamp          DATETIME DEFAULT CURRENT_TIMESTAMP,
            reviewed           INTEGER DEFAULT 0,
            added_to_training  INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_question(question: str, detected_intent: str, confidence: float):
    """Insert an unrecognized question into the database."""
    ensure_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO unknown_questions
                (question, detected_intent, confidence, timestamp)
            VALUES (?, ?, ?, ?)
        """, (question, detected_intent, confidence, datetime.now()))
        conn.commit()
        conn.close()
        print(f"[FallbackLogger] Saved: '{question}' | intent={detected_intent} | conf={confidence:.2f}")
    except Exception as e:
        print(f"[FallbackLogger ERROR] {e}")


class ActionHandleFallback(Action):
    """
    Custom fallback action.
    Triggered when intent confidence < threshold OR intent == nlu_fallback.
    Logs the unknown message to SQLite for later training review.
    """

    def name(self) -> str:
        return "action_handle_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        user_message    = tracker.latest_message.get("text", "")
        intent_name     = tracker.latest_message["intent"]["name"]
        confidence      = tracker.latest_message["intent"]["confidence"]

        print(f"[Fallback] msg='{user_message}' intent='{intent_name}' conf={confidence:.2f}")

        # Log to DB
        log_question(user_message, intent_name, confidence)

        # Reply to user
        dispatcher.utter_message(
            text="I'm sorry, I didn't quite understand that. "
                 "Your question has been recorded and our team will "
                 "use it to improve the system!"
        )
        return []