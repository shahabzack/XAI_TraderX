"""Update Chroma vector store for the trading chatbot.

This module updates a Chroma collection with new embeddings and text chunks for
stock predictions, trades, and balance data for Reliance and Axis Bank.
"""

# Standard library imports
import logging
from logging.handlers import RotatingFileHandler


# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("update_chroma.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def update_chroma(collection, embeddings: list, text_chunks: list) -> None:
    """Update a Chroma collection with new embeddings and text chunks.

    Clears existing data and adds new embeddings, documents, and metadata to the
    collection.

    Args:
        collection: Chroma collection to update.
        embeddings (list): List of embedding vectors.
        text_chunks (list): List of dictionaries with 'text' and 'metadata' keys.

    Returns:
        None

    Raises:
        ValueError: If inputs are empty or invalid.
        TypeError: If inputs have incorrect types or structure.
        RuntimeError: If Chroma operations fail.
    """
    try:
        logger.debug("Updating Chroma collection with %s chunks", len(text_chunks))

        # Validate inputs
        if not text_chunks or not embeddings:
            logger.error("Text chunks or embeddings list is empty")
            raise ValueError("Text chunks and embeddings cannot be empty")
        if len(text_chunks) != len(embeddings):
            logger.error("Mismatched lengths: %s chunks, %s embeddings", len(text_chunks), len(embeddings))
            raise ValueError("Text chunks and embeddings must have the same length")
        if not isinstance(text_chunks, list) or not isinstance(embeddings, (list, tuple)):
            logger.error("Invalid input types: text_chunks=%s, embeddings=%s", type(text_chunks), type(embeddings))
            raise TypeError("Text chunks and embeddings must be lists")
        for chunk in text_chunks:
            if not isinstance(chunk, dict) or "text" not in chunk or "metadata" not in chunk:
                logger.error("Invalid text chunk: %s", chunk)
                raise TypeError("Each text chunk must be a dict with 'text' and 'metadata'")

        # Clear existing data
        logger.debug("Fetching existing IDs")
        try:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                logger.debug("Deleting %s existing IDs", len(existing_ids))
                collection.delete(existing_ids)
        except Exception as e:
            logger.error("Failed to clear existing data")
            raise RuntimeError(f"Error clearing collection: {e}")

        # Add new data
        logger.debug("Adding %s new embeddings and chunks", len(text_chunks))
        try:
            collection.add(
                embeddings=embeddings,
                documents=[chunk["text"] for chunk in text_chunks],
                metadatas=[chunk["metadata"] for chunk in text_chunks],
                ids=[f"doc_{i}" for i in range(len(text_chunks))]
            )
        except Exception as e:
            logger.error("Failed to add new data to collection")
            raise RuntimeError(f"Error adding data to collection: {e}")

        logger.info("Chroma collection updated with %s items", len(text_chunks))

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Chroma update error: %s", e)
        raise
    except Exception as e:
        logger.critical("Unexpected error during Chroma update: %s", e)
        raise