"""Fetch and update stock data for XAI TraderX from MarketStack API.

This script retrieves the latest end-of-day stock data for Reliance and Axis Bank,
inserts it into the stock_data table, and updates actual prices in the daily_predictions
table in the SQLite database.
"""

# Standard library imports
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Third-party imports
import pandas as pd
import requests
import sqlite3

# Configure logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("stock_data.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("MARKETSTACK_API_KEY")
if not API_KEY:
    logger.critical("MARKETSTACK_API_KEY is not set in .env")
    raise ValueError("MARKETSTACK_API_KEY is required")
STOCK_MAP = {"RELIANCE.XNSE": "Reliance", "AXISBANK.XNSE": "Axis Bank"}
DATABASE_PATH = os.getenv("DATABASE_PATH")
if not DATABASE_PATH:
    logger.critical("DATABASE_PATH is not set in .env")
    raise ValueError("DATABASE_PATH is required")



def fetch_new_data_from_marketstack(symbol: str, db_stock_name: str) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch the latest EOD stock data from MarketStack API.

    Args:
        symbol: MarketStack stock symbol (e.g., 'RELIANCE.XNSE').
        db_stock_name: Stock name in the database (e.g., 'Reliance').

    Returns:
        tuple: (DataFrame with stock data, error message if any).

    Raises:
        requests.RequestException: For network or API errors.
    """
    logger.debug("Fetching data for symbol=%s, stock=%s", symbol, db_stock_name)
    url = f"http://api.marketstack.com/v1/eod/latest?access_key={API_KEY}&symbols={symbol}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            error_message = f"No data returned for {symbol}. Response: {data}"
            logger.warning(error_message)
            return None, error_message

        latest = data["data"][0]
        df = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime(latest["date"]).date(),
                    "stock_name": db_stock_name,
                    "open": latest["open"],
                    "high": latest["high"],
                    "low": latest["low"],
                    "close": latest["close"],
                    "volume": latest["volume"],
                }
            ]
        )
        logger.info("Fetched data for %s on %s", db_stock_name, df["date"].iloc[0])
        return df, None

    except requests.HTTPError as e:
        error_message = f"HTTP error fetching data for {symbol}: {e.response.status_code} - {e.response.text}"
        logger.error(error_message)
        return None, error_message
    except requests.RequestException as e:
        error_message = f"Network error fetching data for {symbol}: {str(e)}"
        logger.error(error_message)
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error fetching data for {symbol}: {str(e)}"
        logger.critical(error_message)
        return None, error_message


def insert_new_data_into_stock_table(conn: sqlite3.Connection, df: pd.DataFrame | None) -> None:
    """Insert stock data into the stock_data table, avoiding duplicates.

    Args:
        conn: SQLite database connection.
        df: DataFrame with stock data.

    Raises:
        sqlite3.Error: For database query or insertion errors.
    """
    if df is None or df.empty:
        logger.warning("No stock data to insert")
        return

    logger.debug("Inserting stock data into stock_data table")
    cursor = conn.cursor()
    try:
        for row in df.itertuples():
            date_str = row.date.strftime("%Y-%m-%d")
            cursor.execute(
                """
                SELECT 1 FROM stock_data WHERE date = ? AND stock_name = ?
                """,
                (date_str, row.stock_name),
            )
            exists = cursor.fetchone()

            if not exists:
                cursor.execute(
                    """
                    INSERT INTO stock_data (date, stock_name, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date_str,
                        row.stock_name,
                        row.open,
                        row.high,
                        row.low,
                        row.close,
                        row.volume,
                    ),
                )
                logger.info("Inserted stock data for %s on %s", row.stock_name, date_str)
            else:
                logger.warning("Data for %s on %s already exists", row.stock_name, date_str)

        conn.commit()
        logger.info("Stock data insertion completed")
    except sqlite3.Error as e:
        conn.rollback()
        error_message = (
            "Database insertion failed: Please try again later."
            if "database is locked" in str(e).lower()
            else f"Database error inserting stock data: {str(e)}"
        )
        logger.error(error_message)
        raise sqlite3.Error(error_message)
    except Exception as e:
        conn.rollback()
        logger.critical("Unexpected error inserting stock data: %s", e)
        raise


def update_actual_price_in_predictions(conn: sqlite3.Connection, df: pd.DataFrame | None) -> None:
    """Update actual_price in daily_predictions table with today's close price.

    Args:
        conn: SQLite database connection.
        df: DataFrame with stock data.

    Raises:
        sqlite3.Error: For database query or update errors.
    """
    if df is None or df.empty:
        logger.warning("No data to update predictions")
        return

    logger.debug("Updating actual prices in daily_predictions")
    cursor = conn.cursor()
    try:
        for row in df.itertuples():
            date_str = row.date.strftime("%Y-%m-%d")
            cursor.execute(
                """
                UPDATE daily_predictions
                SET actual_price = ?
                WHERE stock_name = ? AND target_date = ? AND actual_price IS NULL
                """,
                (row.close, row.stock_name, date_str),
            )
            if cursor.rowcount:
                logger.info("Updated actual price for %s on %s", row.stock_name, date_str)
            else:
                logger.info("No matching prediction to update for %s on %s", row.stock_name, date_str)

        conn.commit()
        logger.info("Actual price updates completed")
    except sqlite3.Error as e:
        conn.rollback()
        error_message = (
            "Database update failed: Please try again later."
            if "database is locked" in str(e).lower()
            else f"Database error updating predictions: {str(e)}"
        )
        logger.error(error_message)
        raise sqlite3.Error(error_message)
    except Exception as e:
        conn.rollback()
        logger.critical("Unexpected error updating predictions: %s", e)
        raise


def main():
    """Fetch and update stock data for all configured stocks."""
    logger.info("Starting stock data update process")
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10)
        logger.info("Connected to database: %s", DATABASE_PATH)

        for symbol, db_name in STOCK_MAP.items():
            logger.debug("Processing stock: %s (%s)", db_name, symbol)
            df, error = fetch_new_data_from_marketstack(symbol, db_name)
            if error:
                logger.error("Skipping %s due to fetch error: %s", db_name, error)
                continue
            insert_new_data_into_stock_table(conn, df)
            update_actual_price_in_predictions(conn, df)

        logger.info("All operations completed successfully")
    except sqlite3.Error as e:
        error_message = (
            "Database connection failed: Please try again later."
            if "database is locked" in str(e).lower()
            else f"Database error: {str(e)}"
        )
        logger.critical(error_message)
        raise
    except Exception as e:
        logger.critical("Unexpected error in main: %s", e)
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()