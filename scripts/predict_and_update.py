"""Generate stock price predictions for XAI TraderX using GRU models.

This script fetches 30 days of stock data for Reliance and Axis Bank, uses pre-trained
GRU models with Monte Carlo dropout to predict the next trading day's closing price,
and stores predictions with confidence scores in the SQLite database.
"""

# Standard library imports
import logging
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Tuple, Optional
import sys

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf

# Configure logging to console and file with UTF-8 encoding
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = RotatingFileHandler('/app/logs/predictions.log', maxBytes=1000000, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream handler with UTF-8 encoding
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='strict'))
logger.addHandler(stream_handler)

# Configuration
MODEL_PATHS = {
    'Axis Bank': '/app/models/trained/axisgru30.keras',
    'Reliance': '/app/models/trained/reliancegru30.keras',
}
SCALER_PATHS = {
    'Axis Bank': {
        'X': '/app/models/scalers/scaler_X_axis.pkl',
        'y': '/app/models/scalers/scaler_y_axis.pkl',
    },
    'Reliance': {
        'X': '/app/models/scalers/scaler_X_reliance.pkl',
        'y': '/app/models/scalers/scaler_y_reliance.pkl',
    },
}
DATABASE_PATH = '/app/database/stock_data.db'
SQL_QUERY = """
    SELECT date, high, low, close 
    FROM stock_data 
    WHERE stock_name = ? 
    ORDER BY date DESC 
    LIMIT 30
"""

def enable_dropout_inference(model: tf.keras.Model) -> tf.keras.Model:
    """Enable dropout layers for Monte Carlo inference during prediction.

    Args:
        model: Loaded Keras model.

    Returns:
        tf.keras.Model: Model with dropout enabled for inference.

    Raises:
        ValueError: If model is invalid or cannot be modified.
    """
    logger.debug('Enabling dropout for model inference')
    try:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.trainable = True
        logger.info('Dropout enabled for inference')
        return model
    except Exception as e:
        error_message = f'Failed to enable dropout: {str(e)}'
        logger.critical(error_message)
        raise ValueError(error_message)

def load_models_and_scalers() -> dict:
    """Load GRU models and scalers for both stocks.

    Returns:
        dict: Dictionary containing models and scalers for each stock.

    Raises:
        FileNotFoundError: If model or scaler files are missing.
        tf.errors.OpError: If model loading fails.
    """
    logger.debug('Loading models and scalers')
    resources = {}
    for stock_name in MODEL_PATHS:
        try:
            model = tf.keras.models.load_model(MODEL_PATHS[stock_name],compile=False)
            model = enable_dropout_inference(model)
            scaler_X = joblib.load(SCALER_PATHS[stock_name]['X'])
            scaler_y = joblib.load(SCALER_PATHS[stock_name]['y'])
            resources[stock_name] = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
            }
            logger.info('Loaded model and scalers for %s', stock_name)
        except (FileNotFoundError, tf.errors.OpError) as e:
            error_message = f'Error loading model/scalers for {stock_name}: {str(e)}'
            logger.critical(error_message)
            raise FileNotFoundError(error_message) if isinstance(e, FileNotFoundError) else tf.errors.OpError(error_message)
    return resources

def fetch_stock_data(conn: sqlite3.Connection, stock_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch the last 30 days of stock data from the database.

    Args:
        conn: SQLite database connection.
        stock_name: Stock name ('Reliance' or 'Axis Bank').

    Returns:
        Tuple[pd.DataFrame, Optional[str]]: DataFrame with stock data and error message if any.

    Raises:
        sqlite3.Error: For database query errors.
    """
    logger.debug('Fetching data for %s', stock_name)
    try:
        df = pd.read_sql(SQL_QUERY, conn, params=(stock_name,))
        if df.empty or len(df) < 30:
            error_message = f'Insufficient data for {stock_name}: {len(df)} rows found'
            logger.warning(error_message)
            return pd.DataFrame(), error_message
        latest_date = pd.to_datetime(df['date'].iloc[0]).date()
        logger.info('Fetched %d rows for %s, latest date: %s', len(df), stock_name, latest_date)
        return df, None
    except sqlite3.Error as e:
        error_message = (
            'Database query failed: Please try again later.'
            if 'database is locked' in str(e).lower()
            else f'Database error fetching data for {stock_name}: {str(e)}'
        )
        logger.error(error_message)
        return pd.DataFrame(), error_message
    except Exception as e:
        error_message = f'Unexpected error fetching data for {stock_name}: {str(e)}'
        logger.critical(error_message)
        return pd.DataFrame(), error_message

def prepare_input_data(df: pd.DataFrame, scaler_X: 'joblib.scaler') -> np.ndarray:
    """Prepare input data for GRU model prediction.

    Args:
        df: DataFrame with stock data (date, high, low, close).
        scaler_X: Scaler for input features.

    Returns:
        np.ndarray: Scaled and reshaped input data (1, 30, 3).

    Raises:
        ValueError: If data preparation fails.
    """
    logger.debug('Preparing input data')
    try:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        features = df[['close', 'high', 'low']].copy()
        features.columns = ['Close', 'High', 'Low']
        X_scaled = scaler_X.transform(features)
        X_reshaped = np.reshape(X_scaled, (1, 30, 3))
        logger.info('Input data prepared: shape=%s', X_reshaped.shape)
        return X_reshaped
    except Exception as e:
        error_message = f'Error preparing input data: {str(e)}'
        logger.critical(error_message)
        raise ValueError(error_message)

def predict_with_uncertainty(
    model: tf.keras.Model, X_input: np.ndarray, scaler_y: 'joblib.scaler', n_iterations: int = 100
) -> Tuple[float, float]:
    """Predict stock price with uncertainty using Monte Carlo dropout.

    Args:
        model: Trained GRU model.
        X_input: Scaled input data (1, 30, 3).
        scaler_y: Scaler for output (closing price).
        n_iterations: Number of Monte Carlo iterations.

    Returns:
        Tuple[float, float]: Predicted price and confidence score.

    Raises:
        ValueError: If prediction or inverse scaling fails.
    """
    logger.debug('Predicting with uncertainty: iterations=%d', n_iterations)
    try:
        predictions = []
        for _ in range(n_iterations):
            pred = model(X_input, training=True)
            predictions.append(pred.numpy())
        predictions = np.array(predictions)
        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        predicted_price = float(scaler_y.inverse_transform(mean_prediction)[0][0])
        confidence = float(1 - uncertainty.mean())
        logger.info('Prediction: price=%.2f, confidence=%.2f', predicted_price, confidence)
        return predicted_price, confidence
    except Exception as e:
        error_message = f'Error during prediction: {str(e)}'
        logger.critical(error_message)
        raise ValueError(error_message)

def get_next_market_open_day(current_date: pd.Timestamp) -> pd.Timestamp:
    """Get the next market open day, skipping weekends (Indian market).

    Args:
        current_date: Current date.

    Returns:
        pd.Timestamp: Next market open day.

    Raises:
        ValueError: If date calculation fails.
    """
    logger.debug('Calculating next market open day for %s', current_date)
    try:
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
            next_date += pd.Timedelta(days=1)
        logger.info('Next market open day: %s', next_date)
        return next_date
    except Exception as e:
        error_message = f'Error calculating next market day: {str(e)}'
        logger.critical(error_message)
        raise ValueError(error_message)

def insert_prediction_if_not_exists(
    conn: sqlite3.Connection,
    stock_name: str,
    target_date: pd.Timestamp,
    predicted_price: float,
    confidence_score: float,
) -> None:
    """Insert prediction into daily_predictions table if it doesn't exist.

    Args:
        conn: SQLite database connection.
        stock_name: Stock name.
        target_date: Prediction target date.
        predicted_price: Predicted closing price.
        confidence_score: Confidence score.

    Raises:
        sqlite3.Error: For database insertion errors.
    """
    logger.debug('Inserting prediction for %s on %s', stock_name, target_date)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT 1 FROM daily_predictions
            WHERE stock_name = ? AND target_date = ?
            """,
            (stock_name, target_date.strftime('%Y-%m-%d')),
        )
        if cursor.fetchone():
            logger.warning('Prediction for %s on %s already exists', stock_name, target_date.date())
            return

        cursor.execute(
            """
            INSERT INTO daily_predictions (stock_name, target_date, predicted_price, confidence_score)
            VALUES (?, ?, ?, ?)
            """,
            (
                stock_name,
                target_date.strftime('%Y-%m-%d'),
                predicted_price,
                confidence_score,
            ),
        )
        conn.commit()
        logger.info('Saved prediction for %s on %s', stock_name, target_date.date())
    except sqlite3.Error as e:
        conn.rollback()
        error_message = (
            'Database insertion failed: Please try again later.'
            if 'database is locked' in str(e).lower()
            else f'Database error saving prediction for {stock_name}: {str(e)}'
        )
        logger.error(error_message)
        raise sqlite3.Error(error_message)
    except Exception as e:
        conn.rollback()
        error_message = f'Unexpected error saving prediction for {stock_name}: {str(e)}'
        logger.critical(error_message)
        raise

def main():
    """Generate and store stock price predictions for Reliance and Axis Bank."""
    logger.info('Starting stock price prediction process')
    current_date = pd.Timestamp(datetime.now().date())
    logger.info('Current date: %s', current_date.date())
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10)
        logger.info('Connected to database: %s', DATABASE_PATH)

        resources = load_models_and_scalers()

        for stock_name in MODEL_PATHS:
            logger.debug('Processing predictions for %s', stock_name)
            # Fetch data
            df, error = fetch_stock_data(conn, stock_name)
            if error:
                logger.error('Skipping %s due to data fetch error: %s', stock_name, error)
                print(f"Error fetching data for {stock_name}: {error}")
                continue

            # Check if stock data is up-to-date
            latest_date = pd.to_datetime(df['date'].iloc[0]).date()
            if (current_date.date() - latest_date).days > 2:
                error_message = (
                    f'Stock data for {stock_name} is outdated: latest date is {latest_date}, '
                    f'expected ~{(current_date - pd.Timedelta(days=1)).date()}. Run daily_update.py.'
                )
                logger.error(error_message)
                print(error_message)
                continue

            # Prepare input
            try:
                X_input = prepare_input_data(df, resources[stock_name]['scaler_X'])
            except ValueError as e:
                logger.error('Skipping %s due to input preparation error: %s', stock_name, e)
                print(f"Error preparing input data for {stock_name}: {e}")
                continue

            # Predict
            try:
                predicted_price, confidence = predict_with_uncertainty(
                    resources[stock_name]['model'],
                    X_input,
                    resources[stock_name]['scaler_y'],
                )
            except ValueError as e:
                logger.error('Skipping %s due to prediction error: %s', stock_name, e)
                print(f"Error making prediction for {stock_name}: {e}")
                continue

            # Calculate target date
            latest_date_ts = pd.Timestamp(latest_date)
            target_date = get_next_market_open_day(latest_date_ts)

            # Log and print prediction
            logger.info(
                'Last date %s: %s, Predicted Close for %s: INR %.2f, Confidence: %.2f',
                stock_name,
                latest_date,
                target_date.date(),
                predicted_price,
                confidence,
            )
            print(f"Last date for {stock_name} data: {latest_date.strftime('%Y-%m-%d')}")
            print(f"Predicted Close for {stock_name} on {target_date.strftime('%Y-%m-%d')}: INR {predicted_price:.2f}")
            print(f"Confidence score for {stock_name}: {confidence:.2f}")

            # Save prediction
            insert_prediction_if_not_exists(conn, stock_name, target_date, predicted_price, confidence)

        logger.info('Prediction process completed successfully')
    except (FileNotFoundError, tf.errors.OpError, sqlite3.Error) as e:
        logger.critical('Critical error: %s', e)
        print(f"Critical error: {e}")
        raise
    except Exception as e:
        logger.critical('Unexpected error in main: %s', e)
        print(f"Unexpected error: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info('Database connection closed')

if __name__ == '__main__':
    main()