"""Initialize Chroma vector store for the trading chatbot.

This module sets up a Chroma collection to store embeddings for stock predictions,
trades, and balance data for Reliance and Axis Bank.
"""

# Standard library imports
import logging
from logging.handlers import RotatingFileHandler

# Third-party imports
import chromadb


# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("init_chroma.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def init_chroma(persist_directory: str, collection_name: str):
    """Initialize or retrieve a Chroma collection.

    Creates a persistent Chroma client and returns a collection for storing
    stock data embeddings.

    Args:
        persist_directory (str): Directory path to store Chroma data.
        collection_name (str): Name of the Chroma collection.

    Returns:
        chromadb.Collection: The initialized or existing Chroma collection.

    Raises:
        ValueError: If persist_directory or collection_name is empty.
        RuntimeError: If Chroma client or collection initialization fails.
    """
    try:
        logger.debug("Initializing Chroma collection: %s", collection_name)

        # Validate inputs
        if not persist_directory:
            logger.error("Persist directory is empty")
            raise ValueError("Persist directory cannot be empty")
        if not collection_name:
            logger.error("Collection name is empty")
            raise ValueError("Collection name cannot be empty")

        # Initialize Chroma client
        logger.debug("Creating persistent Chroma client at %s", persist_directory)
        try:
            client = chromadb.PersistentClient(path=persist_directory)
        except Exception as e:
            logger.error("Failed to create Chroma client")
            raise RuntimeError(f"Chroma client initialization error: {e}")

        # Get or create collection
        logger.debug("Getting or creating collection: %s", collection_name)
        try:
            collection = client.get_or_create_collection(name=collection_name)
        except Exception as e:
            logger.error("Failed to get or create collection: %s", collection_name)
            raise RuntimeError(f"Collection initialization error: {e}")

        logger.info("Chroma collection initialized: %s", collection_name)
        return collection

    except (ValueError, RuntimeError) as e:
        logger.error("Chroma initialization error: %s", e)
        raise
    except Exception as e:
        logger.critical("Unexpected error during Chroma initialization: %s", e)
        raise