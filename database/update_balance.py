import sqlite3

DB_PATH = "f:/Xai_traderx/database/stock_data.db"


def add_100_to_balance():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM user_balance WHERE id = 1")
    result = cursor.fetchone()

    if not result:
        cursor.execute("INSERT INTO user_balance (id, balance) VALUES (1, 100100)")
        print("User balance initialized with ₹100100.")
    else:
        current_balance = result[0]
        new_balance = current_balance + 10
        cursor.execute(
            "UPDATE user_balance SET balance = ? WHERE id = 1", (new_balance,)
        )
        print(f"Added ₹100. New balance: ₹{new_balance:.2f}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    add_100_to_balance()
