"""Generate embeddings for stock data text chunks.

This module creates embeddings for text chunks from stock predictions, trades, and
balance data for Reliance and Axis Bank, used by the GRU model chatbot's vector store.
"""

# Standard library imports
import logging
from logging.handlers import RotatingFileHandler

# Third-party imports
from sentence_transformers import SentenceTransformer


# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("generate_embeddings.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_embeddings(
    text_chunks: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32
) -> tuple:
    """Generate embeddings for text chunks using a SentenceTransformer model.

    Converts text chunks from stock data into numerical embeddings for use in
    the chatbot's vector store.

    Args:
        text_chunks (list): List of dictionaries with 'text' and 'metadata' keys.
        model_name (str, optional): Name of the SentenceTransformer model.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        batch_size (int, optional): Batch size for encoding. Defaults to 32.

    Returns:
        tuple: (embeddings, text_chunks) where embeddings is a numpy array of
            encoded vectors and text_chunks is the input list.

    Raises:
        ValueError: If text_chunks is empty or invalid, or batch_size is non-positive.
        TypeError: If text_chunks contains invalid entries.
        RuntimeError: If model loading or encoding fails.
    """
    try:
        logger.debug("Starting embedding generation for %s chunks", len(text_chunks))

        # Validate inputs
        if not text_chunks:
            logger.error("Text chunks list is empty")
            raise ValueError("Text chunks list cannot be empty")
        if not isinstance(text_chunks, list):
            logger.error("Text chunks is not a list")
            raise TypeError("Text chunks must be a list")
        if batch_size <= 0:
            logger.error("Invalid batch size: %s", batch_size)
            raise ValueError("Batch size must be positive")

        # Check text_chunks structure
        for chunk in text_chunks:
            if not isinstance(chunk, dict) or "text" not in chunk or not isinstance(chunk["text"], str):
                logger.error("Invalid text chunk: %s", chunk)
                raise TypeError("Each text chunk must be a dict with a 'text' string")

        # Load model
        logger.debug("Loading SentenceTransformer model: %s", model_name)
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error("Failed to load model: %s", model_name)
            raise RuntimeError(f"Model loading error: {e}")

        # Extract texts
        logger.debug("Extracting texts from chunks")
        texts = [chunk["text"] for chunk in text_chunks]

        # Generate embeddings
        logger.debug("Encoding %s texts with batch size %s", len(texts), batch_size)
        try:
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error("Embedding generation failed")
            raise RuntimeError(f"Embedding encoding error: {e}")

        logger.info("Generated %s embeddings successfully", len(embeddings))
        return embeddings, text_chunks

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Embedding generation error: %s", e)
        raise
    except Exception as e:
        logger.critical("Unexpected error during embedding generation: %s", e)
        raise