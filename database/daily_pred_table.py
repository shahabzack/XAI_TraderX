import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database path from .env
DB_PATH = os.getenv("DATABASE_PATH")

# Ensure the database directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Connect to the database
conn = sqlite3.connect(DB_PATH)  
cursor = conn.cursor()


# Create the daily_predictions table with the updated structure
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS daily_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    predicted_price REAL NOT NULL,
    actual_price REAL,
    confidence_score REAL
);
"""
)

conn.commit()

# List all tables to confirm the table creation
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

conn.close()

print(
    "✅ Table 'daily_predictions' recreated successfully in your existing stock_data.db."
)

# Print all the tables to verify if the new table exists
print("Tables in database:")
for table in tables:
    print("-", table[0])
