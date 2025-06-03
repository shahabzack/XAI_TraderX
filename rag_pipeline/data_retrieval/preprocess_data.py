"""Process stock data for the trading chatbot.

This module converts stock predictions, historical data, trades, and user balance
into text chunks with metadata for Reliance and Axis Bank, used by the GRU model chatbot.
"""

# Standard library imports
import logging
from logging.handlers import RotatingFileHandler

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("preprocess_data.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def preprocess_data(data: dict) -> list:
    """Convert stock data into text chunks with metadata.

    Processes daily predictions, stock data, trades, and user balance into
    formatted text strings with metadata for use in the chatbot's vector store.

    Args:
        data (dict): Dictionary with keys 'daily_predictions', 'stock_data',
            'trades', and 'user_balance' containing database records.

    Returns:
        list: List of dictionaries, each with 'text' (formatted string) and
            'metadata' (table, stock_name, date, etc.).

    Raises:
        ValueError: If data is missing required keys or has invalid format.
        TypeError: If data entries have incorrect types or structure.
    """
    try:
        logger.debug("Starting data preprocessing")
        # Validate input
        if not isinstance(data, dict):
            logger.error("Input data is not a dictionary")
            raise TypeError("Data must be a dictionary")
        required_keys = ["daily_predictions", "stock_data", "trades", "user_balance"]
        if not all(key in data for key in required_keys):
            logger.error("Missing required data keys: %s", required_keys)
            raise ValueError("Data missing required keys: daily_predictions, stock_data, trades, user_balance")

        text_chunks = []

        # Process daily predictions
        logger.debug("Processing daily predictions")
        for pred in data["daily_predictions"]:
            try:
                id, target_date, stock_name, predicted_price, actual_price, confidence_score = pred
                predicted_price = float(predicted_price) if predicted_price is not None else 0.0
                actual_price = float(actual_price) if actual_price is not None else 0.0
                confidence_score = float(confidence_score) if confidence_score is not None else 0.0
                text = (
                    f"Daily Prediction: Stock={stock_name}, Date={target_date}, "
                    f"Predicted Price={predicted_price:.2f}, Actual Price={'None' if not actual_price else f'{actual_price:.2f}'}, "
                    f"Confidence Score={confidence_score:.4f}"
                )
                text_chunks.append({
                    "text": text,
                    "metadata": {
                        "table": "daily_predictions",
                        "stock_name": stock_name,
                        "date": target_date
                    }
                })
            except (ValueError, TypeError) as e:
                logger.warning("Invalid prediction data: %s", pred)
                continue

        # Process stock data
        logger.debug("Processing stock data")
        for stock in data["stock_data"]:
            try:
                date, stock_name, open_price, high_price, low_price, close_price, volume = stock
                open_price = float(open_price) if open_price is not None else 0.0
                high_price = float(high_price) if high_price is not None else 0.0
                low_price = float(low_price) if low_price is not None else 0.0
                close_price = float(close_price) if close_price is not None else 0.0
                volume = int(volume) if volume is not None else 0
                text = (
                    f"Stock Data: Stock={stock_name}, Date={date}, "
                    f"Open={open_price:.2f}, High={high_price:.2f}, Low={low_price:.2f}, "
                    f"Close={close_price:.2f}, Volume={volume}"
                )
                text_chunks.append({
                    "text": text,
                    "metadata": {
                        "table": "stock_data",
                        "stock_name": stock_name,
                        "date": date
                    }
                })
            except (ValueError, TypeError) as e:
                logger.warning("Invalid stock data: %s", stock)
                continue

        # Process trades
        logger.debug("Processing trades")
        for trade in data["trades"]:
            try:
                (
                    id, stock_name, action, entry_time, entry_price, confidence_score,
                    exit_time, exit_price, profit_loss, status, lot_size, trade_type, is_short
                ) = trade
                entry_price = float(entry_price) if entry_price is not None else 0.0
                exit_price = float(exit_price) if exit_price is not None else 0.0
                profit_loss = float(profit_loss) if profit_loss is not None else 0.0
                lot_size = int(lot_size) if lot_size is not None else 0
                text = (
                    f"Trade: Stock={stock_name}, Action={action}, "
                    f"Entry Price={entry_price:.2f}, Exit Price={exit_price:.2f}, "
                    f"Profit/Loss={profit_loss:.2f}, Lot Size={lot_size}, "
                    f"Entry Time={entry_time}, Exit Time={'None' if not exit_time else exit_time}, "
                    f"Status={status}, Is Short={is_short}"
                )
                text_chunks.append({
                    "text": text,
                    "metadata": {
                        "table": "trades",
                        "stock_name": stock_name,
                        "entry_time": entry_time
                    }
                })
            except (ValueError, TypeError) as e:
                logger.warning("Invalid trade data: %s", trade)
                continue

        # Process user balance
        logger.debug("Processing user balance")
        for balance in data["user_balance"]:
            try:
                id, balance_amount = balance
                balance_amount = float(balance_amount) if balance_amount is not None else 0.0
                text = f"User Balance: ID={id}, Amount={balance_amount:.2f}"
                text_chunks.append({
                    "text": text,
                    "metadata": {"table": "user_balance"}
                })
            except (ValueError, TypeError) as e:
                logger.warning("Invalid balance data: %s", balance)
                continue

        logger.info("Processed %s text chunks", len(text_chunks))
        return text_chunks

    except (ValueError, TypeError) as e:
        logger.error("Preprocessing error: %s", e)
        raise
    except Exception as e:
        logger.critical("Unexpected error during preprocessing: %s", e)
        raise