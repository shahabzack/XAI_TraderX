"""Main FastAPI application for the trading chatbot.

This module provides API endpoints for trading, balance queries, stock data, and
chatbot interactions, managing Reliance and Axis Bank stock predictions and trades.
It integrates a Retrieval-Augmented Generation (RAG) pipeline for natural language
query processing and uses SQLite for data persistence.
"""

# Standard library imports
import logging
import os
from datetime import datetime, time, timedelta
from logging.handlers import RotatingFileHandler
import sqlite3
import yaml
from typing import List, Dict, Optional
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Third-party imports
from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
from pydantic import BaseModel,validator
import re

# Local imports
from database.simulation_trade import (
    get_all_trades,
    get_balance,
    get_open_trades,
    init_db,
)
from rag_pipeline.data_retrieval.fetch_data import fetch_all_data
from rag_pipeline.data_retrieval.preprocess_data import preprocess_data
from rag_pipeline.embeddings.generate_embeddings import generate_embeddings
from rag_pipeline.llm.inference import run_inference
from rag_pipeline.llm.load_llm import load_llm
from rag_pipeline.vector_store.init_chroma import init_chroma
from rag_pipeline.vector_store.update_chroma import update_chroma

# Configure logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("main.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Disable oneDNN optimizations for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize FastAPI app
app = FastAPI()

# Database configuration
DB_PATH = os.getenv("DATABASE_PATH")
if not DB_PATH:
    logger.critical("DATABASE_PATH is not set in .env")
    raise ValueError("DATABASE_PATH is required")

# Allowed stock names
ALLOWED_STOCKS = ["Reliance", "Axis Bank"]


def log_buy_trade(
    stock_name: str,
    price: float,
    confidence_score: float,
    lot_size: int,
    trade_type: str,
) -> tuple[bool, str]:
    """Log a buy trade and update user balance."""
    logger.debug("Logging buy trade: stock=%s, price=%s, lot_size=%s", stock_name, price, lot_size)
    conn = None
    try:
        if stock_name not in ALLOWED_STOCKS:
            logger.error("Invalid stock name: %s", stock_name)
            raise ValueError(f"Stock must be one of {ALLOWED_STOCKS}")
        if lot_size <= 0:
            logger.error("Invalid lot size: %s", lot_size)
            raise ValueError("Lot size must be greater than zero")
        if price < 10.0:
            logger.error("Invalid price: %s", price)
            raise ValueError("Price must be >= 10.0")
        if trade_type not in ["intraday", "long-term"]:
            logger.error("Invalid trade type: %s", trade_type)
            raise ValueError("Trade type must be 'intraday' or 'long-term'")

        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute("BEGIN")

        cursor.execute(
            """
            SELECT id FROM trades 
            WHERE stock_name = ? AND trade_type = ? AND status = 'OPEN'
            """,
            (stock_name, trade_type),
        )
        if cursor.fetchone():
            logger.warning("Existing open %s trade for %s", trade_type, stock_name)
            return (
                False,
                f"Cannot open a new {trade_type} trade for {stock_name}: An existing open trade must be closed first.",
            )

        total_cost = price * lot_size
        cursor.execute("SELECT balance FROM user_balance WHERE id = 1")
        result = cursor.fetchone()
        if not result:
            logger.info("No balance found, initializing with INR 100,000")
            cursor.execute("INSERT INTO user_balance (id, balance) VALUES (1, 100000)")
            current_balance = 100000
        else:
            current_balance = result[0]

        if current_balance < total_cost:
            logger.warning("Insufficient funds: required=INR %s, available=INR %s", total_cost, current_balance)
            return (
                False,
                f"Insufficient funds for buy trade. Required: INR {total_cost:.2f}, Available: INR {current_balance:.2f}",
            )

        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO trades (stock_name, action, entry_time, entry_price, confidence_score, lot_size, trade_type, status, is_short)
            VALUES (?, 'BUY', ?, ?, ?, ?, ?, 'OPEN', 0)
            """,
            (stock_name, entry_time, price, confidence_score, lot_size, trade_type),
        )
        trade_id = cursor.lastrowid
        logger.info("Inserted buy trade ID %s: stock=%s, price=%s", trade_id, stock_name, price)

        new_balance = current_balance - total_cost
        cursor.execute("UPDATE user_balance SET balance = ? WHERE id = 1", (new_balance,))
        conn.commit()
        logger.info("Balance updated: new_balance=INR %s", new_balance)

        return (
            True,
            f"Buy order placed for {stock_name}. INR {total_cost:.2f} deducted from balance.",
        )

    except ValueError as e:
        logger.error("Input validation error: %s", e)
        raise
    except sqlite3.Error as e:
        error_message = (
            "Database connection failed: Please try again later."
            if "database is locked" in str(e).lower()
            else f"Database error: {str(e)}"
        )
        logger.exception("Database error in log_buy_trade")
        if conn:
            conn.rollback()
        return False, error_message
    except Exception as e:
        logger.critical("Unexpected error in log_buy_trade: %s", e)
        if conn:
            conn.rollback()
        return False, "Unexpected error occurred"
    finally:
        if conn:
            conn.close()


def log_sell_trade(
    stock_name: str,
    price: float,
    lot_size: int,
    trade_type: str,
    is_short: bool = False,
    is_opening_short: bool = False,
) -> tuple[bool, str]:
    """Log a sell trade and update user balance."""
    logger.debug("Logging sell trade: stock=%s, price=%s, lot_size=%s, is_short=%s", stock_name, price, lot_size, is_short)
    conn = None
    try:
        if stock_name not in ALLOWED_STOCKS:
            logger.error("Invalid stock name: %s", stock_name)
            raise ValueError(f"Stock must be one of {ALLOWED_STOCKS}")
        if lot_size <= 0:
            logger.error("Invalid lot size: %s", lot_size)
            raise ValueError("Lot size must be greater than zero")
        if price < 10.0:
            logger.error("Invalid price: %s", price)
            raise ValueError("Price must be >= 10.0")
        if trade_type not in ["intraday", "long-term"]:
            logger.error("Invalid trade type: %s", trade_type)
            raise ValueError("Trade type must be 'intraday' or 'long-term'")

        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute("BEGIN")

        if is_opening_short and trade_type == "intraday":
            cursor.execute(
                """
                SELECT id FROM trades 
                WHERE stock_name = ? AND trade_type = ? AND status = 'OPEN'
                """,
                (stock_name, trade_type),
            )
            if cursor.fetchone():
                logger.warning("Existing open %s trade for %s", trade_type, stock_name)
                return (
                    False,
                    f"Cannot open a new {trade_type} trade for {stock_name}: An existing open trade must be closed first.",
                )

            margin_required = price * lot_size
            cursor.execute("SELECT balance FROM user_balance WHERE id = 1")
            result = cursor.fetchone()
            if not result:
                logger.info("No balance found, initializing with INR 100,000")
                cursor.execute("INSERT INTO user_balance (id, balance) VALUES (1, 100000)")
                current_balance = 100000
            else:
                current_balance = result[0]

            if current_balance < margin_required:
                logger.warning("Insufficient funds: required=INR %s, available=INR %s", margin_required, current_balance)
                return (
                    False,
                    f"Insufficient funds for short sell. Required: INR {margin_required:.2f}, Available: INR {current_balance:.2f}"
                )

            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                INSERT INTO trades (stock_name, action, entry_time, entry_price, lot_size, trade_type, status, is_short)
                VALUES (?, 'SELL', ?, ?, ?, ?, 'OPEN', 1)
                """,
                (stock_name, entry_time, price, lot_size, trade_type),
            )
            trade_id = cursor.lastrowid
            logger.info("Inserted short sell trade ID %s: stock=%s, price=%s", trade_id, stock_name, price)

            new_balance = current_balance - margin_required
            cursor.execute("UPDATE user_balance SET balance = ? WHERE id = 1", (new_balance,))
            conn.commit()
            logger.info("Balance updated: new_balance=INR %s", new_balance)

            return (
                True,
                f"Short sell order opened for {stock_name}. INR {margin_required:.2f} deducted from balance.",
            )

        cursor.execute(
            """
            SELECT id, entry_price, lot_size, trade_type, is_short 
            FROM trades 
            WHERE stock_name = ? AND trade_type = ? AND status = 'OPEN'
            ORDER BY entry_time ASC
            """,
            (stock_name, trade_type),
        )
        open_trade = cursor.fetchone()
        logger.debug("Queried open trade for %s, %s: %s", stock_name, trade_type, open_trade)
        if not open_trade:
            logger.warning("No open %s trade for %s", trade_type, stock_name)
            return False, f"No open {trade_type} trade found for {stock_name} to close."

        trade_id, entry_price, open_lot_size, open_trade_type, is_short_trade = open_trade
        logger.info("Found trade ID %s: entry_price=%s, lot_size=%s", trade_id, entry_price, open_lot_size)

        if entry_price is None or price is None:
            logger.error("Missing price data: entry_price=%s, exit_price=%s", entry_price, price)
            return False, "Invalid trade data: Missing price information."
        if entry_price < 10.0 or price < 10.0:
            logger.error("Invalid prices: entry_price=%s, exit_price=%s", entry_price, price)
            return (
                False,
                f"Invalid price: entry_price={entry_price}, exit_price={price}. Prices must be >= 10.0.",
            )
        if lot_size > open_lot_size:
            logger.error("Requested lot_size=%s exceeds open_lot_size=%s", lot_size, open_lot_size)
            return (
                False,
                f"Requested lot size {lot_size} exceeds open trade lot size {open_lot_size}.",
            )

        profit_loss = (
            (price - entry_price) * lot_size
            if not is_short_trade
            else (entry_price - price) * lot_size
        )
        total_return = (entry_price * lot_size) + profit_loss
        logger.debug("Calculated: profit_loss=%s, total_return=%s", profit_loss, total_return)

        if lot_size == open_lot_size:
            cursor.execute(
                """
                UPDATE trades 
                SET exit_price = ?, exit_time = ?, profit_loss = ?, status = 'CLOSED'
                WHERE id = ?
                """,
                (
                    price,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    profit_loss,
                    trade_id,
                ),
            )
            logger.info("Full closure for trade ID %s, status=CLOSED", trade_id)
        else:
            new_lot_size = open_lot_size - lot_size
            if new_lot_size <= 0:
                logger.error("Invalid new lot size: %s", new_lot_size)
                return (
                    False,
                    f"Cannot reduce lot size to {new_lot_size}. Lot size must remain greater than zero.",
                )
            cursor.execute(
                """
                UPDATE trades 
                SET lot_size = ?, exit_price = ?, exit_time = ?, profit_loss = ?
                WHERE id = ?
                """,
                (
                    new_lot_size,
                    price,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    profit_loss,
                    trade_id,
                ),
            )
            logger.info("Partial closure for trade ID %s, new_lot_size=%s", trade_id, new_lot_size)

        cursor.execute("SELECT lot_size FROM trades WHERE id = ?", (trade_id,))
        updated_lot_size = cursor.fetchone()[0]
        logger.debug("Verified lot_size for trade_id %s: %s", trade_id, updated_lot_size)
        if updated_lot_size <= 0 and lot_size != open_lot_size:
            logger.error("Invalid lot_size after update: %s", updated_lot_size)
            return False, "Invalid lot size after partial closure"

        cursor.execute("SELECT balance FROM user_balance WHERE id = 1")
        result = cursor.fetchone()
        if not result:
            logger.info("No balance found, initializing with INR 100,000")
            cursor.execute("INSERT INTO user_balance (id, balance) VALUES (1, 100000)")
            current_balance = 100000
        else:
            current_balance = result[0]

        new_balance = current_balance + total_return
        if new_balance <= 0:
            logger.warning("Resetting negative balance to INR 100,000")
            new_balance = 100000
        cursor.execute("UPDATE user_balance SET balance = ? WHERE id = 1", (new_balance,))
        conn.commit()
        logger.info("Balance updated: new_balance=INR %s", new_balance)

        profit_loss_str = (
            f"profit of INR {profit_loss:.2f}" if profit_loss >= 0 else f"loss of INR {-profit_loss:.2f}"
        )
        return (
            True,
            f"Trade closed for {stock_name}. {profit_loss_str.capitalize()} and INR {entry_price * lot_size:.2f} {'margin' if is_short_trade else 'investment'} returned. Total credited: INR {total_return:.2f}.",
        )

    except ValueError as e:
        logger.error("Input validation error: %s", e)
        raise
    except sqlite3.Error as e:
        error_message = (
            "Database connection failed: Please try again later."
            if "database is locked" in str(e).lower()
            else f"Database error: {str(e)}"
        )
        logger.exception("Database error in log_sell_trade")
        if conn:
            conn.rollback()
        return False, error_message
    except Exception as e:
        logger.critical("Unexpected error in log_sell_trade: %s", e)
        if conn:
            conn.rollback()
        return False, "Unexpected error occurred"
    finally:
        if conn:
            conn.close()


# Initialize the database
logger.info("Initializing database")
init_db()


@app.get("/")
async def read_root():
    """Root endpoint for the trading API."""
    logger.debug("Accessing root endpoint")
    return {
        "message": "Welcome to the Trading Simulation API. Use /docs to view the Swagger UI."
    }


class TradeRequest(BaseModel):
    stock_name: str
    action: str
    entry_price: float
    confidence_score: float
    lot_size: int
    trade_type: str
    exit_price: Optional[float] = None

    @validator("stock_name")
    def check_stock_name(cls, v):
        if v not in ALLOWED_STOCKS:
            raise ValueError(f"Invalid stock_name: Must be one of {ALLOWED_STOCKS}")
        return v

    @validator("entry_price")
    def check_entry_price(cls, v):
        if v < 10.0:
            raise ValueError(f"Invalid entry_price: {v}. Must be >= 10.0.")
        return v

    @validator("lot_size")
    def check_lot_size(cls, v):
        if v <= 0:
            raise ValueError("Lot size must be greater than zero.")
        return v


class ExitTradeRequest(BaseModel):
    """Pydantic model for exit trade request data."""
    stock_name: str
    exit_price: float
    trade_type: str
    lot_size: int

    @validator("stock_name")
    def check_stock_name(cls, v):
        if v not in ALLOWED_STOCKS:
            raise ValueError(f"Invalid stock_name: Must be one of {ALLOWED_STOCKS}")
        return v

    @validator("exit_price")
    def check_exit_price(cls, v):
        if v < 10.0:
            raise ValueError(f"Invalid exit_price: {v}. Must be >= 10.0.")
        return v

    @validator("lot_size")
    def check_lot_size(cls, v):
        if v <= 0:
            raise ValueError("Lot size must be greater than zero.")
        return v

@app.get("/balance")
async def get_user_balance():
    """Fetch the current user balance."""
    logger.debug("Fetching user balance")
    try:
        balance = get_balance()
        logger.info("Balance retrieved: INR %s", balance)
        return {"balance": balance}
    except sqlite3.Error as e:
        logger.error("Database error fetching balance: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch balance")


@app.post("/trade/")
async def enter_trade(trade: TradeRequest):
    """Enter a new trade (buy or sell)."""
    logger.debug("Processing trade: stock=%s, action=%s", trade.stock_name, trade.action)
    try:
        if trade.trade_type not in ["intraday", "long-term"]:
            logger.error("Invalid trade_type: %s", trade.trade_type)
            raise HTTPException(
                status_code=400,
                detail="Invalid trade_type: Must be 'intraday' or 'long-term'.",
            )
        if trade.lot_size <= 0:
            logger.error("Invalid lot_size: %s", trade.lot_size)
            raise HTTPException(status_code=400, detail="Lot size must be greater than zero")

        current_time = datetime.now()
        if trade.trade_type == "intraday":
            if current_time.weekday() >= 5:
                logger.warning("Intraday trade attempted on weekend")
                raise HTTPException(
                    status_code=400,
                    detail="Intraday trading is not available on weekends. Try a long-term trade or wait until Monday.",
                )
            if current_time.time() >= time(15, 30):
                logger.warning("Intraday trade attempted after 3:30 PM")
                raise HTTPException(
                    status_code=400,
                    detail="Intraday trading is closed after 3:30 PM. Try a long-term trade or wait until tomorrow.",
                )

        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM trades 
                WHERE stock_name = ? AND trade_type = ? AND status = 'OPEN'
                """,
                (trade.stock_name, trade.trade_type),
            )
            if cursor.fetchone():
                logger.warning("Existing open %s trade for %s", trade.trade_type, trade.stock_name)
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot open a new {trade.trade_type} trade for {trade.stock_name}: An existing open trade must be closed first.",
                )
        except sqlite3.Error as e:
            error_message = (
                "Database connection failed: Please try again later."
                if "database is locked" in str(e).lower()
                else f"Database error: {str(e)}"
            )
            logger.error("Database error in enter_trade: %s", e)
            raise HTTPException(status_code=500, detail=error_message)
        finally:
            if conn:
                conn.close()

        if trade.action == "BUY":
            total_cost = trade.entry_price * trade.lot_size
            if get_balance() < total_cost:
                logger.warning("Insufficient funds: required=INR %s, available=INR %s", total_cost, get_balance())
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient funds for buy trade. Required: INR {total_cost:.2f}, Available: INR {get_balance():.2f}",
                )
            result, message = log_buy_trade(
                trade.stock_name,
                trade.entry_price,
                trade.confidence_score,
                trade.lot_size,
                trade.trade_type,
            )
        elif trade.action == "SELL":
            if trade.trade_type == "intraday":
                result, message = log_sell_trade(
                    trade.stock_name,
                    trade.entry_price,
                    trade.lot_size,
                    trade.trade_type,
                    is_short=True,
                    is_opening_short=True,
                )
            else:
                if trade.exit_price is None:
                    logger.error("Missing exit_price for long-term SELL")
                    raise HTTPException(
                        status_code=400,
                        detail="Exit price must be provided for long-term SELL.",
                    )
                result, message = log_sell_trade(
                    trade.stock_name,
                    trade.exit_price,
                    trade.lot_size,
                    trade.trade_type,
                    is_short=False,
                )
        else:
            logger.error("Invalid action: %s", trade.action)
            raise HTTPException(
                status_code=400, detail="Invalid action: Must be 'BUY' or 'SELL'."
            )

        if result:
            logger.info("Trade successful: %s", message)
            return {"message": message}
        else:
            logger.warning("Trade failed: %s", message)
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Trade input error: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.critical("Unexpected error in enter_trade: %s", e)
        raise HTTPException(status_code=500, detail="Trade processing failed")


@app.post("/exit_trade/")
async def manual_exit_trade(exit_trade: ExitTradeRequest):
    """Manually exit an open trade."""
    logger.debug("Processing exit trade: stock=%s, trade_type=%s", exit_trade.stock_name, exit_trade.trade_type)
    try:
        if exit_trade.trade_type not in ["intraday", "long-term"]:
            logger.error("Invalid trade_type: %s", exit_trade.trade_type)
            raise HTTPException(
                status_code=400,
                detail="Invalid trade_type: Must be 'intraday' or 'long-term'.",
            )
        if exit_trade.lot_size <= 0:
            logger.error("Invalid lot_size: %s", exit_trade.lot_size)
            raise HTTPException(status_code=400, detail="Lot size must be greater than zero")

        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, is_short 
                FROM trades 
                WHERE stock_name = ? AND trade_type = ? AND status = 'OPEN'
                ORDER BY entry_time ASC
                """,
                (exit_trade.stock_name, exit_trade.trade_type),
            )
            open_trade = cursor.fetchone()
            if not open_trade:
                logger.warning("No open %s trade for %s", exit_trade.trade_type, exit_trade.stock_name)
                raise HTTPException(
                    status_code=400,
                    detail=f"No open {exit_trade.trade_type} trade found for {exit_trade.stock_name} to exit.",
                )
            trade_id, is_short = open_trade
            logger.debug("Found open trade ID %s, is_short=%s", trade_id, is_short)
        except sqlite3.Error as e:
            error_message = (
                "Database connection failed: Please try again later."
                if "database is locked" in str(e).lower()
                else f"Database error: {str(e)}"
            )
            logger.error("Database error in exit_trade: %s", e)
            raise HTTPException(status_code=500, detail=error_message)
        finally:
            if conn:
                conn.close()

        result, message = log_sell_trade(
            exit_trade.stock_name,
            exit_trade.exit_price,
            exit_trade.lot_size,
            exit_trade.trade_type,
            is_short=is_short,
            is_opening_short=False,
        )
        logger.info("Exit trade result: %s", message)
        if result:
            return {"message": message}
        else:
            raise HTTPException(status_code=400, detail=message)

    except HTTPException:
        raise
    except Exception as e:
        logger.critical("Unexpected error in exit_trade: %s", e)
        raise HTTPException(status_code=500, detail="Exit trade processing failed")


@app.get("/open_trades")
async def get_open_trades_endpoint():
    """Fetch all open trades."""
    logger.debug("Fetching open trades")
    try:
        open_trades = get_open_trades()
        logger.info("Retrieved %s open trades", len(open_trades))
        return {"open_trades": open_trades}
    except sqlite3.Error as e:
        logger.error("Database error fetching open trades: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch open trades")


@app.get("/trade_history")
async def get_trade_history():
    """Fetch closed trade history."""
    logger.debug("Fetching trade history")
    try:
        trade_history = get_all_trades()
        trade_history = [trade for trade in trade_history if trade["status"] == "CLOSED"]
        logger.info("Retrieved %s closed trades", len(trade_history))
        return {"trade_history": trade_history}
    except sqlite3.Error as e:
        logger.error("Database error fetching trade history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch trade history")


@app.get("/debug_trades")
async def debug_trades():
    """Fetch all trades for debugging purposes."""
    logger.debug("Fetching all trades for debugging")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, stock_name, action, entry_time, entry_price, lot_size, trade_type, status, is_short
            FROM trades 
            ORDER BY entry_time DESC
            """
        )
        rows = cursor.fetchall()
        trades = [
            {
                "id": row[0],
                "stock_name": row[1],
                "action": row[2],
                "entry_time": row[3],
                "entry_price": row[4],
                "lot_size": row[5],
                "trade_type": row[6],
                "status": row[7],
                "is_short": row[8],
            }
            for row in rows
        ]
        logger.info("Retrieved %s trades for debugging", len(trades))
        return {"trades": trades}
    except sqlite3.Error as e:
        error_message = (
            "Database connection failed: Please try again later."
            if "database is locked" in str(e).lower()
            else f"Database error: {str(e)}"
        )
        logger.error("Database error in debug_trades: %s", e)
        raise HTTPException(status_code=500, detail=error_message)
    finally:
        if conn:
            conn.close()


def update_user_balance(amount: float):
    """Update user balance by a specified amount."""
    logger.error("Attempted to use update_user_balance")
    raise NotImplementedError("Use log_buy_trade or log_sell_trade for balance updates within a transaction")




def load_config(config_path: str = "rag_pipeline/config/config.yaml") -> Dict:
    """Load configuration from environment variables."""
    logger.debug("Loading config from environment variables")
    try:
        required_vars = [
            "DATABASE_PATH", "GEMINI_API_KEY", "EMBEDDING_MODEL_NAME",
            "CHROMA_PERSIST_DIRECTORY", "CHROMA_COLLECTION_NAME",
            "SCHEDULER_UPDATE_INTERVAL", "SCHEDULER_UPDATE_TIME"
        ]
        for var in required_vars:
            if not os.getenv(var):
                logger.critical("%s is not set in .env", var)
                raise ValueError(f"{var} is required")
        config = {
            "database": {
                "path": os.getenv("DATABASE_PATH")
            },
            "llm": {
                "model_name": "gemini-1.5-flash",
                "device": "api",
                "api_key": os.getenv("GEMINI_API_KEY")
            },
            "embedding": {
                "model_name": os.getenv("EMBEDDING_MODEL_NAME")
            },
            "chroma": {
                "persist_directory": os.getenv("CHROMA_PERSIST_DIRECTORY"),
                "collection_name": os.getenv("CHROMA_COLLECTION_NAME")
            },
            "scheduler": {
                "update_interval": os.getenv("SCHEDULER_UPDATE_INTERVAL"),
                "update_time": os.getenv("SCHEDULER_UPDATE_TIME")
            }
        }
        logger.info("Successfully loaded configuration from environment variables")
        return config
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load configuration")


# RAGPipeline class
class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for processing stock data and user queries."""

    def __init__(self, config: Dict):
        """Initialize the RAG pipeline with configuration."""
        logger.debug("Initializing RAG pipeline")
        try:
            required_keys = ["embedding", "chroma", "llm", "database"]
            if not all(key in config for key in required_keys):
                logger.error("Missing config keys: %s", required_keys)
                raise ValueError("Configuration missing required keys")

            self.config = config
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=config["embedding"]["model_name"],
            )
            self.collection = init_chroma(
                config["chroma"]["persist_directory"],
                self.config["chroma"]["collection_name"],
            )
            self.model, self.tokenizer = load_llm(
                config["llm"]["model_name"],
                self.config["llm"]["device"],
                self.config["llm"].get("api_key", ""),
            )
            self.vector_store = Chroma(
                collection_name=config["chroma"]["collection_name"],
                embedding_function=self.embedding_model,
                persist_directory=config["chroma"]["persist_directory"],
            )
            self.last_queried_date = None
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize RAG pipeline: %s", e)
            raise RuntimeError("RAG pipeline initialization failed")

    def update_pipeline(self):
        """Update the RAG pipeline with new data."""
        logger.debug("Updating RAG pipeline")
        try:
            data = fetch_all_data(self.config["database"]["path"], days_back=30)
            text_chunks = preprocess_data(data)
            embeddings, text_chunks = generate_embeddings(
                text_chunks, self.config["embedding"]["model_name"]
            )
            update_chroma(self.collection, embeddings, text_chunks)
            logger.info("RAG pipeline updated successfully")
        except Exception as e:
            logger.error("Failed to update RAG pipeline: %s", e)
            raise RuntimeError("Pipeline update failed")

    def get_last_trading_days(self, num_days: int) -> list:
        """Get the last N trading days, skipping weekends."""
        try:
            logger.debug("Fetching last %s trading days", num_days)
            if num_days <= 0:
                logger.error("Invalid number of days: %s", num_days)
                raise ValueError("Number of days must be positive")
            today = datetime.now().date()
            trading_days = []
            days_checked = 0
            offset = 0
            while len(trading_days) < num_days and days_checked < num_days + 10:
                check_date = today - timedelta(days=offset)
                if check_date.weekday() < 5:  # Monday to Friday
                    trading_days.append(check_date.strftime("%Y-%m-%d"))
                offset += 1
                days_checked += 1
            logger.info("Fetched %s trading days", len(trading_days))
            return trading_days
        except Exception as e:
            logger.exception("Failed to fetch trading days")
            raise

    def get_monthly_trade_summary(self, trades: List[Dict], stock_name: Optional[str] = None, target_month: Optional[str] = None) -> Dict[str, Dict]:
        """Calculate monthly trade summaries including profit/loss and trade counts."""
        try:
            logger.debug("Generating monthly trade summary for stock: %s, month: %s", stock_name or "all", target_month or "all")
            monthly_summary = {}
            for trade in trades:
                if stock_name and stock_name.lower() not in trade["stock"].lower():
                    continue
                entry_date = datetime.strptime(trade["entry_date"], "%Y-%m-%d")
                month_key = entry_date.strftime("%Y-%m")
                if target_month and month_key != target_month:
                    continue
                month_name = entry_date.strftime("%B %Y")  # e.g., "May 2025"
                if month_key not in monthly_summary:
                    monthly_summary[month_key] = {
                        "month_name": month_name,
                        "total_trades": 0,
                        "intraday_trades": 0,
                        "long_term_trades": 0,
                        "total_profit": 0.0,
                        "total_loss": 0.0,
                        "net_profit_loss": 0.0,
                        "stock_summary": {},  # Dynamically initialize stock_summary
                        "trades": []
                    }
                stock = trade["stock"]
                if stock not in monthly_summary[month_key]["stock_summary"]:
                    monthly_summary[month_key]["stock_summary"][stock] = {
                        "trades": 0,
                        "profit": 0.0,
                        "loss": 0.0,
                        "net_profit_loss": 0.0
                    }
                monthly_summary[month_key]["total_trades"] += 1
                if trade["trade_type"].lower() == "intraday":
                    monthly_summary[month_key]["intraday_trades"] += 1
                else:
                    monthly_summary[month_key]["long_term_trades"] += 1
                profit_loss = trade["profit_loss"]
                if profit_loss >= 0:
                    monthly_summary[month_key]["total_profit"] += profit_loss
                    monthly_summary[month_key]["stock_summary"][stock]["profit"] += profit_loss
                else:
                    monthly_summary[month_key]["total_loss"] += abs(profit_loss)
                    monthly_summary[month_key]["stock_summary"][stock]["loss"] += abs(profit_loss)
                monthly_summary[month_key]["stock_summary"][stock]["net_profit_loss"] += profit_loss
                monthly_summary[month_key]["stock_summary"][stock]["trades"] += 1
                monthly_summary[month_key]["trades"].append(trade)

            # Fix: update net profit/loss per month here AFTER all trades processed
            for month_key, data in monthly_summary.items():
                data["net_profit_loss"] = data["total_profit"] - data["total_loss"]

            logger.info("Generated summary for %d months", len(monthly_summary))
            return monthly_summary
        except Exception as e:
            logger.exception("Failed to generate monthly trade summary")
            raise ValueError(f"Error in monthly summary: {e}")

    def format_trade_response(
        self,
        trades: List[Dict],
        query_type: str,
        specific_date: Optional[str] = None,
        stock_name: Optional[str] = None,
        days_back: int = None,
        target_month: Optional[str] = None
    ) -> str:
        """Format trade data into a user-friendly response."""
        try:
            logger.debug("Formatting trade response for query_type: %s", query_type)
            if not trades and query_type != "open_trades":
                date_str = f" on {specific_date}" if specific_date else ""
                stock_str = f" for {stock_name}" if stock_name else ""
                month_str = f" for {datetime.strptime(target_month, '%Y-%m').strftime('%B %Y')}" if target_month else ""
                if query_type == "last_x_days" and days_back is not None:
                    return f"No trades found in the last {days_back} days{stock_str}. Want to check recent trades?"
                if query_type == "monthly_summary" and target_month:
                    return f"No trades found{stock_str}{month_str}. Try a different month."
                return f"No trades found{stock_str}{date_str}. Try another date or stock."

            total_profit = sum(t["profit_loss"] for t in trades if t["profit_loss"] is not None and t["profit_loss"] >= 0)
            total_loss = sum(abs(t["profit_loss"]) for t in trades if t["profit_loss"] is not None and t["profit_loss"] < 0)
            net_profit_loss = total_profit - total_loss
            stocks_traded = list(set(t["stock"] for t in trades)) if trades else []
            date_str = f" on {specific_date}" if specific_date else ""
            stock_str = f" for {stock_name}" if stock_name else ""
            month_str = f" for {datetime.strptime(target_month, '%Y-%m').strftime('%B %Y')}" if target_month else ""

            if query_type == "open_trades":
                open_trades = get_open_trades()
                if stock_name:
                    open_trades = [t for t in open_trades if stock_name.lower() in t["stock_name"].lower()]
                if specific_date:
                    open_trades = [t for t in open_trades if t["entry_time"].startswith(specific_date)]
                if not open_trades:
                    return f"No open trades found{stock_str}{date_str}. Want to see closed trades?"
                response = f"Open trades{stock_str}{date_str}: {len(open_trades)} trade{'s' if len(open_trades) > 1 else ''}\n\n"
                for trade in open_trades:
                    response += f"- {trade['stock_name']}: {trade['action']} on {trade['entry_time'].split(' ')[0]}, Lot Size: {trade['lot_size']}, Entry Price: ₹{trade['entry_price']:.2f}, Status: {trade['status']}, Entered: {trade['entry_time']}\n"
                response += "\nWant to see closed trades or predictions?"
                return response

            if query_type == "last_x_days":
                response = f"In the last {days_back} days, you made {len(trades)} trade{'s' if len(trades) > 1 else ''} on {', '.join(stocks_traded)}"
                response += f", earning ₹{total_profit:.2f} in profits, ₹{total_loss:.2f} in losses, net ₹{net_profit_loss:.2f}."
                response += "\n\nDetails:\n"
                for trade in trades:
                    profit_loss_str = f"{'Profit' if trade['profit_loss'] >= 0 else 'Loss'}: ₹{abs(trade['profit_loss']):.2f}"
                    response += f"- {trade['stock']}: {trade['action']} on {trade['entry_date']}, {profit_loss_str}, Lot Size: {trade['lot_size']}\n"
                response += "\nWant more trade details or predictions?"
                return response

            if query_type == "last_one":
                trade = trades[-1]
                profit_loss_str = f"{'Profit' if trade['profit_loss'] >= 0 else 'Loss'}: ₹{abs(trade['profit_loss']):.2f}"
                response = f"Your latest trade{stock_str}{date_str}: {trade['stock']}, {trade['action']} ({trade['trade_type_label']}) on {trade['entry_date']}, {profit_loss_str}, Lot Size: {trade['lot_size']}."
                response += f" Started at {trade['entry_time']}, ended at {trade['exit_time']} (Status: {trade['status']})."
                response += "\n\nWant more trades or stock predictions?"
                return response

            if query_type == "intraday":
                response = f"Your intraday trades{stock_str}{date_str}: {len(trades)} trade{'s' if len(trades) > 1 else ''} on {', '.join(stocks_traded)}."
                response += f" Profits: ₹{total_profit:.2f}, Losses: ₹{total_loss:.2f}, Net: ₹{net_profit_loss:.2f}."
                response += "\n\nDetails:\n"
                for trade in trades:
                    profit_loss_str = f"{'Profit' if trade['profit_loss'] >= 0 else 'Loss'}: ₹{abs(trade['profit_loss']):.2f}"
                    response += f"- {trade['stock']}: {trade['action']} ({trade['trade_type_label']}), {profit_loss_str}, Lot Size: {trade['lot_size']}, Entered: {trade['entry_time']}, Exited: {trade['exit_time']}\n"
                response += "\nWant specific trade details or predictions?"
                return response

            if query_type == "profit":
                if not trades:
                    return f"No trades found{stock_str}{date_str}. Try another date or stock."
                monthly_summary = self.get_monthly_trade_summary(trades, stock_name)
                response = f"Your total trades{stock_str}{date_str}: {len(trades)} trade{'s' if len(trades) > 1 else ''} ({sum(t['trade_type'].lower() == 'intraday' for t in trades)} intraday, {sum(t['trade_type'].lower() == 'long-term' for t in trades)} long-term)."
                response += f"\nTotal Profit: ₹{total_profit:.2f}, Total Loss: ₹{total_loss:.2f}, Net Profit/Loss: ₹{net_profit_loss:.2f}"
                response += "\n\nStock-wise Net Profit/Loss:\n"
                stock_summary = {}
                for trade in trades:
                    stock = trade["stock"]
                    if stock not in stock_summary:
                        stock_summary[stock] = {"net": 0.0}
                    stock_summary[stock]["net"] += trade["profit_loss"]
                for stock, data in sorted(stock_summary.items()):
                    response += f"- {stock}: ₹{data['net']:.2f}\n"
                response += "\nMonthly Net Profit/Loss:\n"
                for month_key, data in sorted(monthly_summary.items()):
                    response += f"- {data['month_name']}: ₹{data['net_profit_loss']:.2f}\n"
                response += "\nWant details for a specific month or stock predictions?"
                return response

            if query_type == "loss":
                loss_trades = [t for t in trades if t["profit_loss"] < 0]
                if not loss_trades:
                    return f"No losing trades found{stock_str}{date_str}. Want to check profitable trades?"
                monthly_summary = self.get_monthly_trade_summary(trades, stock_name)
                response = f"Your losing trades{stock_str}{date_str}: ₹{total_loss:.2f} from {len(loss_trades)} trade{'s' if len(loss_trades) > 1 else ''}."
                response += "\n\nStock-wise Net Loss:\n"
                stock_summary = {}
                for trade in loss_trades:
                    stock = trade["stock"]
                    if stock not in stock_summary:
                        stock_summary[stock] = {"net_loss": 0.0}
                    stock_summary[stock]["net_loss"] += abs(trade["profit_loss"])
                for stock, data in sorted(stock_summary.items()):
                    response += f"- {stock}: ₹{data['net_loss']:.2f}\n"
                response += "\nMonthly Net Loss:\n"
                for month_key, data in sorted(monthly_summary.items()):
                    response += f"- {data['month_name']}: ₹{data['total_loss']:.2f}\n"
                response += "\nWant details for a specific month or stock predictions?"
                return response

            if query_type == "trade_dates":
                trade_dates = sorted(set(t["entry_date"] for t in trades))
                response = f"Your trade dates{stock_str}:\n"
                for date in trade_dates:
                    response += f"- {date}\n"
                response += "\nWant details for any of these days?"
                return response

            if query_type == "trade_count":
                intraday_count = len([t for t in trades if t["trade_type"].lower() == "intraday"])
                long_term_count = len([t for t in trades if t["trade_type"].lower() == "long-term"])
                response = f"You made {len(trades)} trade{'s' if len(trades) != 1 else ''} on {stock_name or 'Reliance and Axis Bank'}{date_str}: "
                response += f"{intraday_count} intraday, {long_term_count} long-term."
                response += "\nWant trade details?"
                return response

            if query_type == "stock_profit_loss":
                stock_summary = {}
                for trade in trades:
                    stock = trade["stock"]
                    if stock not in stock_summary:
                        stock_summary[stock] = {"profit": 0.0, "loss": 0.0, "net_profit_loss": 0.0, "trades": 0}
                    stock_summary[stock]["trades"] += 1
                    if trade["profit_loss"] >= 0:
                        stock_summary[stock]["profit"] += trade["profit_loss"]
                    else:
                        stock_summary[stock]["loss"] += abs(trade["profit_loss"])
                    stock_summary[stock]["net_profit_loss"] += trade["profit_loss"]
                
                max_profit_stock = max(stock_summary.items(), key=lambda x: x[1]["profit"], default=("None", {"profit": 0}))
                max_loss_stock = max(stock_summary.items(), key=lambda x: x[1]["loss"], default=("None", {"loss": 0}))
                response = f"Profit/Loss by stock{date_str}:\n"
                for stock, data in stock_summary.items():
                    response += f"- {stock}: {data['trades']} trades, Profit: ₹{data['profit']:.2f}, Loss: ₹{data['loss']:.2f}, Net: ₹{data['net_profit_loss']:.2f}\n"
                response += f"\nMost profitable stock: {max_profit_stock[0]} (₹{max_profit_stock[1]['profit']:.2f})\n"
                response += f"Most loss-making stock: {max_loss_stock[0]} (₹{max_loss_stock[1]['loss']:.2f})\n"
                response += "\nWant detailed trade history?"
                return response

            if query_type == "monthly_summary":
                monthly_summary = self.get_monthly_trade_summary(trades, stock_name, target_month)
                if not monthly_summary:
                    return f"No trades found{stock_str}{month_str or ' in recent months'}. Try a different period."
                response = f"Your monthly trade summary{stock_str}{month_str}:\n\n"
                for month_key, data in sorted(monthly_summary.items()):
                    response += f"{data['month_name']}:\n"
                    response += f"- Total Trades: {data['total_trades']} ({data['intraday_trades']} intraday, {data['long_term_trades']} long-term)\n"
                    response += f"- Total Profit: ₹{data['total_profit']:.2f}\n"
                    response += f"- Total Loss: ₹{data['total_loss']:.2f}\n"
                    response += f"- Net Profit/Loss: ₹{data['net_profit_loss']:.2f}\n"
                    response += f"- Stock-wise Breakdown:\n"
                    for stock, stock_data in sorted(data['stock_summary'].items()):
                        if stock_data['trades'] > 0:
                            response += f"  - {stock}: {stock_data['trades']} trades, Profit: ₹{stock_data['profit']:.2f}, Loss: ₹{stock_data['loss']:.2f}, Net: ₹{stock_data['net_profit_loss']:.2f}\n"
                    response += "\n"
                if len(monthly_summary) > 1:
                    max_profit_month = max(monthly_summary.items(), key=lambda x: x[1]["total_profit"], default=("None", {"total_profit": 0}))
                    max_loss_month = max(monthly_summary.items(), key=lambda x: x[1]["total_loss"], default=("None", {"total_loss": 0}))
                    if max_profit_month[0] != "None":
                        response += f"Most profitable month: {max_profit_month[1]['month_name']} (₹{max_profit_month[1]['total_profit']:.2f})\n"
                    if max_loss_month[0] != "None":
                        response += f"Most loss-making month: {max_loss_month[1]['month_name']} (₹{max_loss_month[1]['total_loss']:.2f})\n"
                response += "\nWant detailed trades for a specific month or stock?"
                return response

            if query_type == "today_trades":
                closed_trades = trades
                open_trades = get_open_trades()
                if stock_name:
                    closed_trades = [t for t in closed_trades if stock_name.lower() in t["stock"].lower()]
                    open_trades = [t for t in open_trades if stock_name.lower() in t["stock_name"].lower()]
                if specific_date:
                    closed_trades = [t for t in closed_trades if t["entry_date"] == specific_date]
                    open_trades = [t for t in open_trades if t["entry_time"].startswith(specific_date)]
                response = f"Trades today{stock_str}{date_str}:\n"
                if closed_trades:
                    response += f"Closed Trades: {len(closed_trades)} trade{'s' if len(closed_trades) > 1 else ''}\n"
                    for trade in closed_trades:
                        profit_loss_str = f"{'Profit' if trade['profit_loss'] >= 0 else 'Loss'}: ₹{abs(trade['profit_loss']):.2f}"
                        response += f"- {trade['stock']}: {trade['action']} ({trade['trade_type_label']}), {profit_loss_str}, Lot Size: {trade['lot_size']}, Entered: {trade['entry_time']}, Exited: {trade['exit_time']}\n"
                else:
                    response += "No closed trades today.\n"
                if open_trades:
                    response += f"\nOpen Trades: {len(open_trades)} trade{'s' if len(open_trades) > 1 else ''}\n"
                    for trade in open_trades:
                        response += f"- {trade['stock_name']}: {trade['action']}, Lot Size: {trade['lot_size']}, Entry Price: ₹{trade['entry_price']:.2f}, Status: {trade['status']}, Entered: {trade['entry_time']}\n"
                else:
                    response += "\nNo open trades today.\n"
                response += f"\nTotal Closed Profit: ₹{total_profit:.2f}, Loss: ₹{total_loss:.2f}, Net: ₹{net_profit_loss:.2f}"
                response += "\nWant more trade details or predictions?"
                return response

            trades_to_show = trades[-4:] if len(trades) >= 4 else trades
            response = f"Your recent trades{stock_str}{date_str}: {len(trades)} trade{'s' if len(trades) > 1 else ''} on {', '.join(stocks_traded) if stocks_traded else 'none'}."
            response += f" Profits: ₹{total_profit:.2f}, Losses: ₹{total_loss:.2f}, Net: ₹{net_profit_loss:.2f}."
            response += f"\n\nLatest {len(trades_to_show)} trades:\n" if len(trades) > 4 else "\n\nTrades:\n"
            for trade in trades_to_show:
                profit_loss_str = f"{'Profit' if trade['profit_loss'] >= 0 else 'Loss'}: ₹{abs(trade['profit_loss']):.2f}"
                response += f"- {trade['stock']}: {trade['action']} on {trade['entry_date']}, {profit_loss_str}, Lot Size: {trade['lot_size']}, Status: {trade['status']}, Entered: {trade['entry_time']}, Exited: {trade['exit_time']}\n"
            response += "\nWant specific stock or date details?"
            return response
        except Exception as e:
            logger.exception("Failed to format trade response")
            raise ValueError(f"Invalid query_type or data: {e}")

    def get_latest_prediction_date(self) -> Optional[str]:
        """Get the latest date for stock predictions."""
        logger.debug("Fetching latest prediction date")
        for attempt in range(3):  # Retry logic
            conn = None
            try:
                conn = sqlite3.connect(self.config["database"]["path"], timeout=20)  # Increased timeout
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT target_date 
                    FROM daily_predictions 
                    WHERE target_date IS NOT NULL 
                    AND target_date LIKE '____-__-__' 
                    ORDER BY target_date DESC 
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if not row:
                    logger.warning("No valid predictions found in daily_predictions")
                    return None
                latest_date = row[0]
                try:
                    datetime.strptime(latest_date, "%Y-%m-%d")
                    logger.info("Latest prediction date: %s", latest_date)
                    return latest_date
                except ValueError:
                    logger.error("Invalid date format in database: %s", latest_date)
                    return None
            except sqlite3.Error as e:
                logger.error("Database error fetching latest prediction date (attempt %d): %s", attempt + 1, e)
                if attempt == 2:
                    raise RuntimeError(f"Failed to fetch latest prediction date: {str(e)}")
            finally:
                if conn:
                    conn.close()

    def query(self, user_query: str) -> str:
        """Process a user query and generate a response."""
        try:
            logger.debug("Processing query: %s", user_query)
            days_back = 30
            specific_date = None
            target_month = None
            user_query_lower = user_query.lower().strip()

            # Extended typo corrections
            typo_corrections = {
                "pasta": "past", "predicted": "prediction", "predction": "prediction", "predicition": "prediction",
                "forcast": "forecast", "tarde": "trade", "tad": "trade", "trd": "trade", "prfit": "profit",
                "otmmwor": "tomorrow", "tommorow": "tomorrow", "tommrow":"tomorrow","yestr": "yesterday", "rleicen": "reliance",
                "releicn": "reliance", "azisa": "axis", "axisa": "axis", "clso eprioce": "close price",
                "clso": "close", "prioce": "price", "profut": "profit", "proift": "profit", "prfuta": "profit",
                "prfoit": "profit", "lsos": "loss", "lssos": "loss", "ndlsos": "loss", "intdayu": "intraday",
                "intrday": "intraday", "long-temri": "long-term", "longtm": "long-term", "montly": "monthly",
                "amontly": "monthly", "mont": "month", "thsimonth": "this month", "thismonth": "this month",
                "proftio": "portfolio", "portflo": "portfolio", "prtflo": "portfolio", "giveme": "give me",
                "giv": "give", "pmro": "more", "countan": "count", "tiatal": "total", "juen": "june",
                "mro ethen": "more than", "sovl": "solve", "sumary": "summary", "sumamry": "summary",
                "perfom": "performance", "actuly": "actually", "trsesol": "threshold", "rmeht": "remove",
                "trhrshodl": "threshold", "baed": "based", "teh": "the", "eteh": "the", "ot": "to",
                "rmiznie": "minimize", "afte": "after", "pnl": "p/l", "dsy": "day", "stcoka": "stock",
                "speic": "specific", "totla": "total", "myc": "my", "hatbot": "chatbot", "rpesne": "response",
                "reuslt": "result", "otehriwse": "otherwise", "othe": "other", "rrepsone": "response"
            }
            for typo, correct in typo_corrections.items():
                user_query_lower = user_query_lower.replace(typo, correct)

            # Handle greetings or vague queries
            if any(phrase in user_query_lower for phrase in ["hey", "hi", "hello", "what can you do"]):
                logger.debug("Handling greeting or vague query")
                if "trade" in user_query_lower or "past trade" in user_query_lower:
                    user_query_lower = "show my recent trades"
                elif "open" in user_query_lower:
                    user_query_lower = "show open trades"
                elif "prediction" in user_query_lower or "forecast" in user_query_lower:
                    user_query_lower = "latest stock predictions"
                elif any(term in user_query_lower for term in ["profit", "loss", "portfolio", "this month"]):
                    user_query_lower = "monthly trade summary"
                else:
                    return "Hi! I can predict stock prices for Reliance or Axis Bank, show your trade history (including open trades), analyze profits/losses, or check your balance. Try asking 'Show open trades', 'What's Reliance's price tomorrow?', or 'Show my trade summary'."

            # Handle date-specific queries
            if "today" in user_query_lower:
                specific_date = datetime.now().strftime("%Y-%m-%d")
                self.last_queried_date = specific_date
            elif "yesterday" in user_query_lower:
                yesterday = datetime.now() - timedelta(days=1)
                while yesterday.weekday() >= 5:  # Skip weekends
                    yesterday -= timedelta(days=1)
                specific_date = yesterday.strftime("%Y-%m-%d")
                self.last_queried_date = specific_date
            elif "tomorrow" in user_query_lower:
                tomorrow = datetime.now() + timedelta(days=1)
                while tomorrow.weekday() >= 5:  # Skip weekends
                    tomorrow += timedelta(days=1)
                specific_date = tomorrow.strftime("%Y-%m-%d")
                self.last_queried_date = specific_date
            elif "that day" in user_query_lower or "that i asked" in user_query_lower:
                specific_date = self.last_queried_date or datetime.now().strftime("%Y-%m-%d")
            else:
                match_date = re.search(r"(\d{4}/\d{2}/\d{2})|(\d{1,2}/\d{1,2}/\d{4})", user_query_lower)
                if match_date:
                    try:
                        date_str = match_date.group(1) or match_date.group(2)
                        date_format = "%Y/%m/%d" if "/" in date_str and len(date_str.split("/")[0]) == 4 else "%d/%m/%Y"
                        specific_date = datetime.strptime(date_str, date_format).strftime("%Y-%m-%d")
                        self.last_queried_date = specific_date
                    except ValueError:
                        logger.warning("Invalid date format in query: %s", user_query)
                        return "Date format looks off. Use yyyy/mm/dd or dd/mm/yyyy."

            days_of_week = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
            }
            for day_name, day_offset in days_of_week.items():
                if day_name in user_query_lower:
                    today = datetime.now()
                    days_until_target = (day_offset - today.weekday()) % 7 or 7
                    specific_date = (today - timedelta(days=days_until_target)).strftime("%Y-%m-%d")
                    self.last_queried_date = specific_date
                    break

            # Handle month-specific queries
            month_map = {
                "january": "01", "jan": "01", "february": "02", "feb": "02", "march": "03", "mar": "03",
                "april": "04", "apr": "04", "may": "05", "june": "06", "jun": "06",
                "july": "07", "jul": "07", "august": "08", "aug": "08", "september": "09", "sep": "09",
                "october": "10", "oct": "10", "november": "11", "nov": "11", "december": "12", "dec": "12"
            }
            current_year = datetime.now().strftime("%Y")  # e.g., "2025"
            # Check for specific month first
            for month_name, month_num in month_map.items():
                if month_name in user_query_lower:
                    target_month = f"{current_year}-{month_num}"  # e.g., "2025-05" for May
                    logger.debug("Setting target_month to: %s", target_month)
                    break
            # Then check for "this month" or "last month" if no specific month is found
            if not target_month:
                if any(phrase in user_query_lower for phrase in ["this month", "current month"]):
                    target_month = datetime.now().strftime("%Y-%m")  # e.g., "2025-06"
                    logger.debug("Setting target_month to current month: %s", target_month)
                elif any(phrase in user_query_lower for phrase in ["last month", "previous month"]):
                    last_month = (datetime.now().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")  # e.g., "2025-05"
                    target_month = last_month
                    logger.debug("Setting target_month to last month: %s", target_month)

            # Handle time period queries
            match_period = re.search(r"last (\d+) day[s]?", user_query_lower)
            match_weekly = re.search(r"(week\w*|weekly)", user_query_lower)
            match_monthly = re.search(r"(month\w*|monthly|1 month|june)", user_query_lower)
            if match_period:
                days_back = int(match_period.group(1))
                logger.debug("Setting days_back to %d", days_back)
            elif match_weekly:
                days_back = 7
                logger.debug("Setting days_back to 7 for weekly query")
            elif match_monthly and not target_month and "june" not in user_query_lower:
                target_month = datetime.now().strftime("%Y-%m")  # Default to current month
                logger.debug("Setting target_month to current month for monthly query: %s", target_month)

            # Fetch trade data
            logger.debug("Fetching data for trade query")
            data = fetch_all_data(self.config["database"]["path"], days_back=days_back, specific_date=specific_date)
            if not any([data.get("trades"), data.get("daily_predictions"), data.get("user_balance")]):
                logger.warning("No data found for query: %s", user_query)
                month_str = datetime.strptime(target_month, "%Y-%m").strftime("%B %Y") if target_month else "this period"
                return f"No trades or predictions found for {specific_date or month_str}. Try another period or date."

            # Identify stock
            stock_name = None
            stock_keywords = {"reliance": "Reliance", "axis": "Axis Bank"}
            for keyword, actual_name in stock_keywords.items():
                if keyword in user_query_lower:
                    stock_name = actual_name
                    break

            # Handle balance queries
            if "balance" in user_query_lower:
                balance = data["user_balance"][0][1] if data.get("user_balance") else 0
                logger.info("Balance query: ₹%.2f", balance)
                return f"Your balance is ₹{float(balance):,.2f}. Ready to trade?"

            # Handle prediction queries
            if any(term in user_query_lower for term in ["prediction", "forecast", "latest prediction", "close price"]):
                logger.debug("Handling prediction query")
                latest_date = self.get_latest_prediction_date()
                formatted_latest_date = datetime.strptime(latest_date, "%Y-%m-%d").strftime("%B %d, %Y") if latest_date else "unknown"
                
                if not stock_name:
                    return f"Latest predictions for {formatted_latest_date}. Pick a stock (Reliance or Axis Bank) or date. Check recent news on Moneycontrol (https://www.moneycontrol.com) before trading."
                
                # Check if specific_date is a weekend
                is_weekend = False
                if specific_date:
                    try:
                        query_date = datetime.strptime(specific_date, "%Y-%m-%d")
                        if query_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
                            is_weekend = True
                            formatted_date = query_date.strftime("%B %d, %Y")
                    except ValueError:
                        logger.warning("Invalid specific_date format: %s", specific_date)
                        return "Invalid date format. Use 'today', 'tomorrow', or 'YYYY-MM-DD'."

                if specific_date:
                    for pred in data.get("daily_predictions", []):
                        id_, target_date, db_stock_name, predicted_price, actual_price, confidence_score = pred
                        if target_date == specific_date and stock_name.lower() in db_stock_name.lower():
                            formatted_date = datetime.strptime(specific_date, "%Y-%m-%d").strftime("%B %d, %Y")
                            actual_price_str = f"Actual price: ₹{actual_price:.2f}" if actual_price is not None else "No actual price yet"
                            return (
                                f"{stock_name} on {formatted_date}: Predicted price ₹{predicted_price:.2f} "
                                f"({confidence_score * 100:.1f}% confidence). {actual_price_str}. "
                                f"Check recent news on Moneycontrol(https://www.moneycontrol.com) before trading. Want trade details?"
                            )
                    formatted_date = datetime.strptime(specific_date, "%Y-%m-%d").strftime("%B %d, %Y")
                    weekend_msg = " (weekend, markets closed)" if is_weekend else ""
                    return (
                        f"No prediction available for {stock_name} on {formatted_date}{weekend_msg}. "
                        f"The last prediction available is for {formatted_latest_date}. "
                        f"Check back later or try another date."
                    )
                
                # Fallback to latest prediction
                for pred in sorted(data.get("daily_predictions", []), key=lambda x: x[1], reverse=True):
                    id_, target_date, db_stock_name, predicted_price, actual_price, confidence_score = pred
                    if stock_name.lower() in db_stock_name.lower():
                        formatted_date = datetime.strptime(target_date, "%Y-%m-%d").strftime("%B %d, %Y")
                        actual_price_str = f"Actual price: ₹{actual_price:.2f}" if actual_price is not None else "No actual price yet"
                        return (
                            f"Latest {stock_name} prediction for {formatted_date}: Predicted price ₹{predicted_price:.2f} "
                            f"({confidence_score * 100:.1f}% confidence). {actual_price_str}. "
                            f"Check recent news on Moneycontrol(https://www.moneycontrol.com) before trading. Want trade details?"
                        )
                    return (
                        f"No predictions available for {stock_name}. The last prediction is for {formatted_latest_date}. "
                        f"Check Moneycontrol for updates or try another stock."
                    )

            # Handle trade queries
            trades = self._process_trades(data.get("trades", []))
            if stock_name:
                trades = [t for t in trades if stock_name.lower() in t["stock"].lower()]
            if target_month:
                trades = [t for t in trades if t["entry_date"].startswith(target_month)]
            trades.sort(key=lambda x: x["entry_time"])

            # Prioritize monthly summary for profit/loss queries with month context
            if any(term in user_query_lower for term in ["profit", "loss", "portfolio", "trade"]) and \
            any(term in user_query_lower for term in ["month", "monthly", "this month"]):
                if not target_month:
                    target_month = datetime.now().strftime("%Y-%m")  # Default to current month
                return self.format_trade_response(trades, "monthly_summary", specific_date, stock_name, target_month=target_month)

            # Handle open trades query
            if "open" in user_query_lower:
                return self.format_trade_response([], "open_trades", specific_date, stock_name)

            # Handle today's trades
            if "today" in user_query_lower and any(term in user_query_lower for term in ["trade", "trades"]):
                return self.format_trade_response(trades, "today_trades", specific_date, stock_name)

            if match_period and any(term in user_query_lower for term in ["trade", "profit", "loss", "portfolio"]):
                num_days = int(match_period.group(1))
                last_trading_days = self.get_last_trading_days(num_days)
                if not last_trading_days:
                    return f"No trades found for the last {num_days} days. Want recent trades?"
                filtered_trades = [
                    t for t in trades if t["entry_date"] in last_trading_days and (not stock_name or stock_name.lower() in t["stock"].lower())
                ]
                if "profit" in user_query_lower:
                    return self.format_trade_response(filtered_trades, "profit", specific_date, stock_name, days_back=days_back)
                if "loss" in user_query_lower:
                    return self.format_trade_response(filtered_trades, "loss", specific_date, stock_name, days_back=days_back)
                return self.format_trade_response(filtered_trades, "last_x_days", specific_date, stock_name, days_back=days_back)

            if "last one trade" in user_query_lower or "last trade" in user_query_lower:
                return self.format_trade_response([trades[-1]] if trades else [], "last_one", specific_date, stock_name)
            if "date i trade" in user_query_lower:
                return self.format_trade_response(trades, "trade_dates", specific_date, stock_name)
            if any(term in user_query_lower for term in ["how many trades", "total trades", "trade count"]):
                return self.format_trade_response(trades, "trade_count", specific_date, stock_name)
            if "intraday" in user_query_lower:
                intraday_trades = [t for t in trades if t["trade_type"].lower() == "intraday"]
                return self.format_trade_response(intraday_trades, "intraday", specific_date, stock_name)
            if any(term in user_query_lower for term in ["profit", "loss"]) and any(term in user_query_lower for term in ["stock", "which stock", "more profit"]):
                return self.format_trade_response(trades, "stock_profit_loss", specific_date, stock_name)
            if "profit" in user_query_lower:
                return self.format_trade_response(trades, "profit", specific_date, stock_name)
            if "loss" in user_query_lower:
                return self.format_trade_response(trades, "loss", specific_date, stock_name)
            return self.format_trade_response(trades, "general", specific_date, stock_name)
        except Exception as e:
            logger.exception("Failed to process query: %s", str(e))
            return "Sorry, something went wrong processing your query. Please try again or rephrase."

    def _process_trades(self, trade_data: List[tuple]) -> List[Dict]:
        """Process trade data into a structured format."""
        trades = []
        for trade in trade_data:
            try:
                trade_id, stock_name_db, action, entry_time, entry_price, confidence_score, exit_time, exit_price, profit_loss, status, lot_size, trade_type, is_short = trade
                if profit_loss is None or lot_size is None:
                    logger.warning("Skipping trade with missing data: %s", trade)
                    continue
                entry_date = entry_time.split(" ")[0]
                if status == "CLOSED":
                    profit_loss_value = float(profit_loss)
                    trade_type_label = "Short" if is_short else "Buy"
                    trades.append({
                        "stock": stock_name_db,
                        "action": action,
                        "entry_date": entry_date,
                        "profit_loss": profit_loss_value,
                        "lot_size": int(lot_size),
                        "entry_time": entry_time,
                        "exit_time": exit_time or "None",
                        "status": status,
                        "trade_type": trade_type,
                        "trade_type_label": trade_type_label
                    })
            except (ValueError, TypeError) as e:
                logger.warning("Invalid trade data: %s, error: %s", trade, e)
                continue
        return trades



@app.post("/chat")
async def chat(request: dict):
    """Handle chatbot queries for trading information."""
    try:
        user_query = request.get("query", "")
        if not user_query:
            logger.error("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        response = rag_pipeline.query(user_query)
        logger.info("Chat response generated for query: %s", user_query)
        return {"response": response}
    except ValueError as e:
        logger.error("Invalid query: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    except Exception as e:
        logger.critical("Chat processing failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process chat query")

@app.get("/stock_data/{stock_name}")
async def get_stock_data(stock_name: str):
    """Fetch recent stock data for a given stock."""
    if stock_name not in ALLOWED_STOCKS:
        logger.error("Invalid stock name: %s", stock_name)
        raise HTTPException(status_code=400, detail=f"Invalid stock: {stock_name}")
    logger.debug("Fetching stock data for %s", stock_name)
    for attempt in range(3):  # Retry logic
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20)  # Increased timeout
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT date, close_price 
                FROM stock_data 
                WHERE stock_name = ? 
                ORDER BY date DESC 
                LIMIT 30
                """,
                (stock_name,),
            )
            rows = cursor.fetchall()
            if not rows:
                logger.info("No stock data found for %s", stock_name)
                return {"dates": [], "close_prices": []}
            rows.reverse()
            dates = [row[0] for row in rows]
            close_prices = [row[1] for row in rows]
            logger.info("Retrieved %d stock data points for %s", len(dates), stock_name)
            return {"dates": dates, "close_prices": close_prices}
        except sqlite3.Error as e:
            error_message = (
                "Database connection failed: Please try again later."
                if "database is locked" in str(e).lower()
                else f"Database error: {str(e)}"
            )
            logger.error("Database error fetching stock data (attempt %d): %s", attempt + 1, e)
            if attempt == 2:
                raise HTTPException(status_code=500, detail=error_message)
        finally:
            if conn:
                conn.close()

@app.get("/predict/{stock_name}")
async def get_daily_prediction(stock_name: str):
    """Fetch daily stock predictions for a given stock."""
    if stock_name not in ALLOWED_STOCKS:
        logger.error("Invalid stock name: %s", stock_name)
        raise HTTPException(status_code=400, detail=f"Invalid stock: {stock_name}")
    logger.debug("Fetching predictions for %s", stock_name)
    for attempt in range(3):  # Retry logic
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20)  # Increased timeout
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT target_date, predicted_price 
                FROM daily_predictions 
                WHERE stock_name = ? 
                ORDER BY target_date DESC 
                LIMIT 7
                """,
                (stock_name,),
            )
            rows = cursor.fetchall()
            if not rows:
                logger.warning("No predictions found for %s", stock_name)
                return {"predictions": []}
            rows.reverse()
            predictions = [{"target_date": row[0], "predicted_price": row[1]} for row in rows]
            logger.info("Retrieved %d predictions for %s", len(predictions), stock_name)
            return {"predictions": predictions}
        except sqlite3.Error as e:
            error_message = (
                "Database connection failed: Please try again later."
                if "database is locked" in str(e).lower()
                else f"Database error: {str(e)}"
            )
            logger.error("Database error fetching predictions (attempt %d): %s", attempt + 1, e)
            if attempt == 2:
                raise HTTPException(status_code=500, detail=error_message)
        finally:
            if conn:
                conn.close()

@app.get("/predict/{stock_name}/{stock_date}")
async def get_prediction_details(stock_name: str, stock_date: str):
    """Fetch prediction details for a specific stock and date."""
    try:
        datetime.strptime(stock_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format: %s", stock_date)
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    if stock_name not in ALLOWED_STOCKS:
        logger.error("Invalid stock name: %s", stock_name)
        raise HTTPException(status_code=400, detail=f"Invalid stock: {stock_name}")
    logger.debug("Fetching prediction for %s on %s", stock_name, stock_date)
    for attempt in range(3):  # Retry logic
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20)  # Increased timeout
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT target_date, predicted_price, actual_price, confidence_score 
                FROM daily_predictions 
                WHERE stock_name = ? AND target_date = ?
                """,
                (stock_name, stock_date),
            )
            row = cursor.fetchone()
            if not row:
                logger.warning("No prediction found for %s on %s", stock_name, stock_date)
                raise HTTPException(
                    status_code=404,
                    detail=f"No prediction found for {stock_name} on {stock_date}",
                )
            logger.info("Prediction retrieved for %s on %s", stock_name, stock_date)
            return {
                "target_date": row[0],
                "predicted_price": row[1],
                "actual_price": row[2],
                "confidence_score": row[3],
            }
        except sqlite3.Error as e:
            error_message = (
                "Database connection failed: Please try again later."
                if "database is locked" in str(e).lower()
                else f"Database error: {str(e)}"
            )
            logger.error("Database error fetching prediction (attempt %d): %s", attempt + 1, e)
            if attempt == 2:
                raise HTTPException(status_code=500, detail=error_message)
        finally:
            if conn:
                conn.close()

# Initialize RAG pipeline
try:
    logger.debug("Loading configuration")
    config = load_config()
    rag_pipeline = RAGPipeline(config)
    logger.info("RAG pipeline initialized")
except Exception as e:
    logger.critical("Failed to initialize RAG pipeline: %s", e)
    raise RuntimeError("Application startup failed")