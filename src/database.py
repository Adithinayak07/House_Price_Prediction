import sqlite3
import os

# Store DB inside src directory
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area REAL,
            rent REAL,
            locality TEXT,
            BHK INTEGER,
            facing TEXT,
            parking TEXT,
            bathrooms INTEGER,
            prediction REAL,
            model_version TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()