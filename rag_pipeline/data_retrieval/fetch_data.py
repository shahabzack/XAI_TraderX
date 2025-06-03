"""Fetch stock data for the trading chatbot.

This module retrieves stock predictions, historical data, trades, and user balance
from a SQLite database for Reliance and Axis Bank, used by the GRU model chatbot.
"""

# Standard library imports
import logging
import sqlite3
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("fetch_data.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_all_data(db_path: str, days_back: int = 30, specific_date: str = None) -> dict:
    """Fetch stock predictions, trades, and balance from the database.

    Retrieves data for Reliance and Axis Bank, including daily predictions,
    historical stock data, trades, and user balance, for a given time range
    or specific date.

    Args:
        db_path (str): Path to the SQLite database file.
        days_back (int, optional): Number of past days to fetch data for.
            Defaults to 30.
        specific_date (str, optional): Specific date to fetch data for
            (YYYY-MM-DD). Defaults to None.

    Returns:
        dict: Data containing daily_predictions, stock_data, trades, and
            user_balance. Returns empty lists for each key on error.

    Raises:
        ValueError: If db_path is empty or days_back is negative.
        sqlite3.Error: If database operations fail.
    """
    try:
        logger.debug("Fetching data from database: %s", db_path)
        # Validate inputs
        if not db_path:
            logger.error("Database path is empty")
            raise ValueError("Database path cannot be empty")
        if days_back < 0:
            logger.error("Days back is negative: %s", days_back)
            raise ValueError("Days back cannot be negative")

        # Connect to database
        logger.debug("Connecting to SQLite database")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Set date range
        if specific_date:
            try:
                # Validate date format
                datetime.strptime(specific_date, "%Y-%m-%d")
                start_date = end_date = specific_date
                logger.debug("Fetching data for specific date: %s", specific_date)
            except ValueError as e:
                logger.error("Invalid date format: %s", specific_date)
                raise ValueError("Specific date must be in YYYY-MM-DD format")
        else:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            logger.debug("Fetching data from %s to %s", start_date, end_date)

        # Fetch daily predictions
        logger.debug("Querying daily predictions")
        if specific_date:
            cursor.execute(
                """
                SELECT id, target_date, stock_name, predicted_price, actual_price, confidence_score 
                FROM daily_predictions 
                WHERE target_date = ? AND stock_name IN ('Axis Bank', 'Reliance')
                """,
                (specific_date,)
            )
        else:
            cursor.execute(
                """
                SELECT id, target_date, stock_name, predicted_price, actual_price, confidence_score 
                FROM daily_predictions 
                WHERE target_date BETWEEN ? AND ? AND stock_name IN ('Axis Bank', 'Reliance')
                """,
                (start_date, end_date)
            )
        daily_predictions = cursor.fetchall()
        logger.debug("Fetched %d predictions: %s", len(daily_predictions), daily_predictions)

        # Fetch historical stock data
        logger.debug("Querying stock data")
        cursor.execute(
            """
            SELECT date, stock_name, open, high, low, close, volume 
            FROM stock_data 
            WHERE date BETWEEN ? AND ? AND stock_name IN ('Axis Bank', 'Reliance')
            """,
            (start_date, end_date)
        )
        stock_data = cursor.fetchall()

        # Fetch trades
        logger.debug("Querying trades")
        cursor.execute(
            """
            SELECT id, stock_name, action, entry_time, entry_price, confidence_score, 
                   exit_time, exit_price, profit_loss, status, lot_size, trade_type, is_short 
            FROM trades 
            WHERE entry_time BETWEEN ? AND ? AND stock_name IN ('Axis Bank', 'Reliance')
            """,
            (f"{start_date} 00:00:00", f"{end_date} 23:59:59")
        )
        trades = cursor.fetchall()

        # Fetch user balance
        logger.debug("Querying user balance")
        cursor.execute("SELECT id, balance FROM user_balance")
        user_balance = cursor.fetchall()

        # Close connection
        conn.close()
        logger.info("Data fetched successfully: %s predictions, %s stock data, %s trades, %s balances",
                    len(daily_predictions), len(stock_data), len(trades), len(user_balance))

        return {
            "daily_predictions": daily_predictions,
            "stock_data": stock_data,
            "trades": trades,
            "user_balance": user_balance
        }

    except ValueError as e:
        logger.error("Input validation error: %s", e)
        raise
    except sqlite3.Error as e:
        logger.exception("Database error occurred")
        return {
            "daily_predictions": [],
            "stock_data": [],
            "trades": [],
            "user_balance": []
        }
    except Exception as e:
        logger.critical("Unexpected error while fetching data: %s", e)
        return {
            "daily_predictions": [],
            "stock_data": [],
            "trades": [],
            "user_balance": []
        }