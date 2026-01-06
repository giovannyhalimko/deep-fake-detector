import sqlite3
import datetime
import os

DB_NAME = "predictions.db"

def init_db():
    """Initialize the predictions database if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction REAL,
            label TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(filename, prediction, label):
    """Save a prediction result to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO history (filename, prediction, label, timestamp) VALUES (?, ?, ?, ?)',
              (filename, prediction, label, datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_history(limit=50):
    """Retrieve the most recent prediction history."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # To access columns by name
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'filename': row['filename'],
            'prediction': row['prediction'],
            'label': row['label'],
            'timestamp': row['timestamp']
        })
    return history
