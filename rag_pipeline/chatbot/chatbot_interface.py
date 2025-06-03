"""Chatbot interface for stock price predictions and trade info.

This script runs a chatbot that uses a GRU model to predict next-day stock prices
for stocks like Axis Bank and Reliance. Users can ask about tomorrow's prices,
recent trades, or their balance.
"""

# Standard library imports
import argparse
import logging
import os
import sys
import warnings
from logging.handlers import RotatingFileHandler

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("chatbot.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hide TensorFlow and other warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Quiet TensorFlow logs

# Add project root to path
sys.path.append(os.path.abspath("F:/Xai_traderx"))

# Local imports
from rag_pipeline.chatbot.run_chatbot import RAGPipeline, load_config


class ChatbotError(Exception):
    """Custom error for chatbot issues."""
    pass


def run_chatbot(update_pipeline: bool = False) -> None:
    """Run the stock prediction chatbot.

    Loads the GRU model pipeline, updates it if needed, and lets users ask about
    stock prices (e.g., Reliance, Axis Bank), recent trades, or their balance.
    Type 'exit' to quit.

    Args:
        update_pipeline (bool): If True, updates the pipeline with new data.
            Defaults to False.

    Returns:
        None

    Raises:
        ChatbotError: If config loading, pipeline setup, or queries fail.
        Exception: For unexpected errors.

    Example:
        >>> run_chatbot(update_pipeline=True)
        Updating pipeline...
        Done updating.
        Ask about stock prices, trades, or balance. Type 'exit' to quit.
        You: What's Reliance's price tomorrow?
        Chatbot: Predicted price: 3050.75
    """
    try:
        # Load config file Minimally Invasive Species
        logger.debug("Loading config file")
        try:
            config = load_config()
        except Exception as e:
            logger.exception("Config load failed")
            raise ChatbotError(f"Config error: {e}")

        # Set up GRU model pipeline
        logger.debug("Starting RAG pipeline")
        try:
            pipeline = RAGPipeline(config)
        except Exception as e:
            logger.exception("Pipeline setup failed")
            raise ChatbotError(f"Pipeline error: {e}")

        # Update pipeline data if requested
        if update_pipeline:
            logger.info("Updating pipeline with new data...")
            try:
                pipeline.update_pipeline()
                logger.info("Pipeline updated")
                print("Updating pipeline...")
                print("Done updating.")
            except Exception as e:
                logger.exception("Pipeline update failed")
                raise ChatbotError(f"Update error: {e}")

        # Chatbot interaction loop
        logger.info("Chatbot ready. Ask about stocks, trades, or balance. Type 'exit' to quit.")
        print("Ask about stock prices, trades, or balance. Type 'exit' to quit.")
        while True:
            user_query = input("You: ")
            logger.debug("User asked: %s", user_query)
            if user_query.lower() == "exit":
                logger.info("User exited chatbot")
                break
            try:
                response = pipeline.query(user_query)
                logger.info("Response: %s", response)
                print(f"Chatbot: {response}")
            except Exception as e:
                logger.exception("Query failed: %s", user_query)
                raise ChatbotError(f"Query error: {e}")
    except ChatbotError as e:
        logger.error("Chatbot error: %s", e)
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical("Unexpected error: %s", e)
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command-line args and start chatbot
    parser = argparse.ArgumentParser(description="Stock Prediction Chatbot")
    parser.add_argument("--update", action="store_true", help="Update pipeline data")
    args = parser.parse_args()

    # Set TensorFlow environment settings
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_LOGGING_VERBOSITY"] = "0"

    logger.info("Starting Stock Prediction Chatbot")
    run_chatbot(update_pipeline=args.update)