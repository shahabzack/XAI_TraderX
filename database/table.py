import sqlite3

# Path to your database
db_path = "F:/Xai_traderx/database/stock_data.db"

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List of tables to check
tables = ["daily_predictions", "user_balance", "trades", "stock_data"]

# Print structure of each table
for table in tables:
    print(f"\n=== Structure of table: {table} ===")
    try:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        if columns:
            for col in columns:
                col_id, name, dtype, notnull, default_val, pk = col
                print(f" - {name} ({dtype}) {'[PK]' if pk else ''}")
        else:
            print(" - No columns found or table does not exist.")
    except Exception as e:
        print(f"Error reading table {table}: {e}")

# Close the connection
conn.close()
