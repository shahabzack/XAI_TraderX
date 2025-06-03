import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database path from .env
DB_PATH = os.getenv("DATABASE_PATH")

def delete_trades_by_id(trade_ids):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create a string of placeholders like (?, ?) for the IN clause
    placeholders = ', '.join(['?'] * len(trade_ids))
    query = f"DELETE FROM trades WHERE id IN ({placeholders})"

    cursor.execute(query, trade_ids)
    conn.commit()
    conn.close()
    print(f"Deleted trades with IDs: {trade_ids}")

if __name__ == "__main__":
    delete_trades_by_id([89])
