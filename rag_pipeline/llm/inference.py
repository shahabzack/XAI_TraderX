"""Run inference for the trading chatbot.

This module handles text generation using the LLM for user queries about stock
predictions, trades, or balance for Reliance and Axis Bank.
"""

# Standard library imports
import logging
from logging.handlers import RotatingFileHandler


# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("inference.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_inference(model, tokenizer, prompt: str) -> str:
    """Generate a response using the LLM for a given prompt.

    Args:
        model: The loaded LLM model or callable (e.g., API query function).
        tokenizer: The tokenizer (None for API-based models).
        prompt (str): The input prompt for the LLM.

    Returns:
        str: The generated response or an error message if inference fails.

    Raises:
        ValueError: If the prompt is empty or invalid.
        RuntimeError: If model inference fails.
    """
    try:
        logger.debug("Running inference for prompt")
        # Validate input
        if not isinstance(prompt, str) or not prompt.strip():
            logger.error("Invalid prompt: empty or not a string")
            raise ValueError("Prompt must be a non-empty string")

        # Run inference
        logger.debug("Calling model for inference")
        response = model(prompt)
        logger.info("Inference completed successfully")
        return response

    except ValueError as e:
        logger.error("Inference input error: %s", e)
        raise
    except Exception as e:
        logger.exception("Inference failed")
        return "Error generating response."