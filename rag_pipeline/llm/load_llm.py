"""Load the LLM for the trading chatbot.

This module initializes the Gemini API-based LLM for generating responses to
user queries about Reliance and Axis Bank stock predictions, trades, or balance.
"""

# Standard library imports
import logging
from logging.handlers import RotatingFileHandler

# Third-party imports
import requests


# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("llm.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_llm(model_name: str = "gemini-1.5-flash", device: str = "api", api_key: str = None) -> tuple:
    """Load the Gemini API-based LLM for inference.

    Args:
        model_name (str, optional): Name of the Gemini model. Defaults to "gemini-1.5-flash".
        device (str, optional): Device type, must be "api". Defaults to "api".
        api_key (str, optional): Gemini API key. Required.

    Returns:
        tuple: (query_llm, None) where query_llm is a callable for API queries and None is a placeholder for tokenizer.

    Raises:
        ValueError: If api_key is missing or device is not "api".
        RuntimeError: If the API request setup fails.
    """
    try:
        logger.debug("Loading LLM: model=%s, device=%s", model_name, device)

        # Validate inputs
        if not api_key:
            logger.error("API key is missing")
            raise ValueError("Gemini API key required")
        if device != "api":
            logger.error("Invalid device: %s", device)
            raise ValueError("Only 'api' device supported")
        if not model_name:
            logger.error("Model name is empty")
            raise ValueError("Model name cannot be empty")

        # Define API query function
        def query_llm(prompt: str) -> str:
            """Query the Gemini API with a prompt.

            Args:
                prompt (str): The input prompt for the LLM.

            Returns:
                str: The generated response text.

            Raises:
                requests.RequestException: If the API request fails.
                KeyError: If the response format is invalid.
            """
            try:
                logger.debug("Sending API request for prompt")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                response = requests.post(url, json=data)
                response.raise_for_status()
                result = response.json()
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                logger.info("API query successful")
                return response_text
            except requests.RequestException as e:
                logger.error("API request failed: %s", e)
                raise
            except KeyError as e:
                logger.error("Invalid API response format: %s", e)
                raise

        logger.info("LLM loaded successfully: %s", model_name)
        return query_llm, None

    except ValueError as e:
        logger.error("LLM loading error: %s", e)
        raise
    except Exception as e:
        logger.critical("Unexpected error loading LLM: %s", e)
        raise RuntimeError(f"Failed to load LLM: {e}")