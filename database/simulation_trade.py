import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_name TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            confidence_score REAL,
            exit_time TEXT,
            exit_price REAL,
            profit_loss REAL,
            status TEXT NOT NULL,
            lot_size INTEGER NOT NULL,
            trade_type TEXT NOT NULL,
            is_short INTEGER DEFAULT 0
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_balance (
            id INTEGER PRIMARY KEY,
            balance REAL NOT NULL
        )
    """
    )

    cursor.execute("SELECT balance FROM user_balance WHERE id = 1")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO user_balance (id, balance) VALUES (1, 100000)")

    conn.commit()
    conn.close()


def update_user_balance(amount):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM user_balance WHERE id = 1")
    result = cursor.fetchone()
    if not result:
        cursor.execute("INSERT INTO user_balance (id, balance) VALUES (1, 100000)")
        current_balance = 100000
    else:
        current_balance = result[0]

    new_balance = current_balance + amount
    if new_balance <= 0:
        new_balance = 100000  # reset safeguard

    cursor.execute("UPDATE user_balance SET balance = ? WHERE id = 1", (new_balance,))
    conn.commit()
    conn.close()
    return new_balance


def get_balance():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM user_balance WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    if not result:
        update_user_balance(0)
        return 100000
    return result[0]


def log_buy_trade(stock_name, entry_price, confidence_score, lot_size, trade_type):
    total_cost = entry_price * lot_size
    current_balance = get_balance()
    if current_balance < total_cost:
        raise ValueError("Insufficient balance for buy trade.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO trades (stock_name, action, entry_time, entry_price, confidence_score, lot_size, trade_type, status)
        VALUES (?, 'BUY', ?, ?, ?, ?, ?, 'OPEN')
    """,
        (stock_name, entry_time, entry_price, confidence_score, lot_size, trade_type),
    )
    conn.commit()
    conn.close()

    update_user_balance(-total_cost)


def log_sell_trade(stock_name, exit_price, lot_size, trade_type, is_short=False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Handle new short-sell trade (entry)
    if trade_type == "intraday" and is_short:
        margin_required = exit_price * lot_size
        current_balance = get_balance()
        if current_balance < margin_required:
            conn.close()
            return False, "Insufficient balance for short sell (margin needed)."

        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO trades (stock_name, action, entry_time, entry_price, lot_size, trade_type, status, is_short)
            VALUES (?, 'SELL', ?, ?, ?, ?, 'OPEN', 1)
        """,
            (stock_name, entry_time, exit_price, lot_size, trade_type),
        )
        conn.commit()
        conn.close()

        update_user_balance(-margin_required)  # Deduct margin at short-sell time
        return (
            True,
            f"Short sell order placed for {stock_name}. ₹{margin_required} margin deducted.",
        )

    # Close existing trade
    query = """
        SELECT id, entry_price, lot_size, is_short, action
        FROM trades
        WHERE stock_name = ? AND status = 'OPEN' AND trade_type = ?
    """
    if trade_type == "long-term":
        query += " AND action = 'BUY'"

    cursor.execute(query, (stock_name, trade_type))
    trade = cursor.fetchone()

    if not trade:
        conn.close()
        return False, f"No open trade found for {stock_name}."

    trade_id, entry_price, open_lot_size, is_short_trade, action = trade

    if lot_size != open_lot_size:
        conn.close()
        return False, "Partial exits not allowed. Lot size must match open trade."

    # Profit/Loss Calculation & Balance Update
    if is_short_trade:
        profit_loss = (entry_price - exit_price) * open_lot_size
        total_return = entry_price * open_lot_size + profit_loss  # return margin + P&L
        update_user_balance(total_return)
    else:
        profit_loss = (exit_price - entry_price) * open_lot_size
        total_return = entry_price * open_lot_size + profit_loss
        update_user_balance(total_return)

    exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        UPDATE trades
        SET exit_time = ?, exit_price = ?, profit_loss = ?, status = 'CLOSED'
        WHERE id = ?
    """,
        (exit_time, exit_price, profit_loss, trade_id),
    )

    conn.commit()
    conn.close()
    return True, f"Closed trade for {stock_name} with P&L: ₹{profit_loss:.2f}"


def get_open_trades():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
    columns = [desc[0] for desc in cursor.description]
    trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return trades


def get_all_trades():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades")
    columns = [desc[0] for desc in cursor.description]
    trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return trades
