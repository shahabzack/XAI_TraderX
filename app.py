"""Streamlit frontend for XAI TraderX trading chatbot.

This application provides a user interface for trading Reliance and Axis Bank stocks,
displaying stock data, predictions, trade history, and interacting with a chatbot
powered by a FastAPI backend.
"""

# Standard library imports
import json
import logging
import os
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Third-party imports
import pandas as pd
import plotly.graph_objs as go
import requests
import sqlite3
import streamlit as st

# Configure logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("streamlit.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# FastAPI backend URL
API_URL = os.getenv("API_URL")
if not API_URL:
    logger.critical("API_URL is not set in .env")
    raise ValueError("API_URL is required")

# Define trade password 
TRADE_PASSWORD = os.getenv("TRADE_PASSWORD")
if not TRADE_PASSWORD:
    logger.critical("TRADE_PASSWORD is not set in .env")
    raise ValueError("TRADE_PASSWORD is required")

# Database path
DATABASE_PATH = os.getenv("DATABASE_PATH")
if not DATABASE_PATH:
    logger.critical("DATABASE_PATH is not set in .env")
    raise ValueError("DATABASE_PATH is required")

def configure_page():
    """Configure Streamlit page settings and apply custom CSS.

    Raises:
        RuntimeError: If CSS file loading fails.
    """
    logger.debug("Configuring Streamlit page")
    try:
        st.set_page_config(
            page_title="XAI TraderX",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        # Apply custom CSS
        css = """
        /* Global styling */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background-color: #0c1425;
            color: #e6e6e6;
            padding: 1rem;
        }
        h1, h2, h3 {
            color: #fff;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .card {
            background: linear-gradient(145deg, #151f38, #1a2540);
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            padding: 1rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid;
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-3px);
        }
        .card-balance {
            border-left-color: #4CAF50;
        }
        .card-trade {
            border-left-color: #2196F3;
        }
        .card-performance {
            border-left-color: #FF9800;
        }
        .card-sell {
            border-left-color: #f44336;
        }
        .card-buy {
            border-left-color: #4CAF50;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.8rem;
        }
        .card-title {
            font-size: 0.9rem;
            margin: 0;
            color: #cfd8dc;
        }
        .card-value {
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0.5rem 0;
            color: #fff;
        }
        .trade-form {
            background: #151f38;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #2196F3, #4CAF50);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1976D2, #388E3C);
            box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
        }
        .trade-history-container {
            max-height: 300px;
            overflow-y: auto;
            padding: 0.5rem;
            width: 100%;
            border-radius: 10px;
            background: #151f38;
            box-sizing: border-box;
        }
        .trade-history-container::-webkit-scrollbar {
            width: 6px;
        }
        .trade-history-container::-webkit-scrollbar-track {
            background: #151f38;
            border-radius: 10px;
        }
        .trade-history-container::-webkit-scrollbar-thumb {
            background: #2196F3;
            border-radius: 10px;
        }
        .trade-history-container::-webkit-scrollbar-thumb:hover {
            background: #1976D2;
        }
        .chat-container {
        background: #151f38;
        border-radius: 15px;
        padding: 1rem;
        min-height: 800px;
        max-height: 800px;
        overflow-y: auto;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        width: 100%;
    }
        .chat-message-user {
            background: linear-gradient(145deg, #4CAF50, #2E7D32);
            color: white;
            padding: 0.8rem 1rem;
            border-radius: 15px 15px 0 15px;
            margin: 0.8rem 1rem 0.8rem auto;
            max-width: 80%;
            align-self: flex-end;
            box-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
            word-wrap: break-word;
        }
        .chat-message-bot {
            background: linear-gradient(145deg, #2196F3, #1976D2);
            color: white;
            padding: 0.8rem 1rem;
            border-radius: 15px 15px 15px 0;
            margin: 0.8rem auto 0.8rem 1rem;
            max-width: 80%;
            align-self: flex-start;
            box-shadow: 0 2px 10px rgba(33, 150, 243, 0.3);
            word-wrap: break-word;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.6rem;
            background-color: #151f38;
            border-radius: 10px;
            padding: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.5rem;
            border-radius: 8px;
            color: #90a4ae;
            background-color: transparent;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(33, 150, 243, 0.2) !important;
            color: #2196F3 !important;
            font-weight: 600;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-open {
            background-color: #4CAF50;
        }
        .status-closed {
            background-color: #f44336;
        }
        .profit {
            color: #4CAF50;
            font-weight: 600;
        }
        .loss {
            color: #f44336;
            font-weight: 600;
        }
        .ticker {
            font-size: 1.2rem;
            font-weight: 700;
            color: #2196F3;
        }
        input[type="text"], input[type="number"], input[type="password"] {
            background-color: #1a2540;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: white;
            padding: 0.8rem 1rem;
        }
        select {
            background-color: #1a2540;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: white;
            padding: 0.8rem 1rem;
        }
        .stSlider {
            padding: 1rem 0;
        }
        .stSlider [data-baseweb="slider"] {
            height: 0.5rem;
        }
        .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
            background: #2196F3;
            color: white;
            font-weight: 600;
            padding: 0.3rem 0.6rem;
            border-radius: 5px;
        }
        .market-data {
            display: flex;
            justify-content: space-between;
            background: #151f38;
            padding: 0.8rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .market-item {
            text-align: center;
        }
        .market-label {
            font-size: 0.8rem;
            color: #90a4ae;
        }
        .market-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
        }
        .market-change-up {
            color: #4CAF50;
            font-size: 0.9rem;
        }
        .market-change-down {
            color: #f44336;
            font-size: 0.9rem;
        }
        .section-header {
            font-size: 1.2rem;
            font-weight: 500;
            color: #fff;
            margin: 1rem 0 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.5rem;
        }
        * {
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }
        """
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        logger.info("Page configuration and CSS applied")
    except Exception as e:
        logger.critical("Failed to configure page: %s", e)
        raise RuntimeError(f"Page configuration failed: {str(e)}")


def init_session_state():
    """Initialize Streamlit session state for chat history."""
    logger.debug("Initializing session state")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Chat history initialized")

def check_backend_health() -> bool:
    """Check if FastAPI backend is running.

    Returns:
        bool: True if backend is reachable, False otherwise.
    """
    logger.debug("Checking backend health")
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        response.raise_for_status()
        logger.info("Backend health check: OK")
        return True
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.error("Backend health check failed: %s", e)
        return False


def make_api_request(endpoint: str, method: str = "GET", data: dict | None = None) -> tuple[dict | None, str | None]:
    """Make an API request to the FastAPI backend with retries.

    Args:
        endpoint: API endpoint (e.g., '/balance').
        method: HTTP method ('GET' or 'POST').
        data: Payload for POST requests.

    Returns:
        tuple: (response_data, error_message) where response_data is the JSON response or None,
               and error_message is the error string or None.
    """
    logger.debug("Making API request: endpoint=%s, method=%s, data=%s", endpoint, method, data)
    
    if not check_backend_health():
        logger.error("Backend not responding for endpoint: %s", endpoint)
        return None, "Server is not responding. Please try again later."

    try:
        with st.spinner("Processing..."):
            headers = {"Content-Type": "application/json"}
            url = f"{API_URL}{endpoint}"
            
            # Configure retries
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=0.5, 
                          status_forcelist=[500, 502, 503, 504])
            session.mount("http://", HTTPAdapter(max_retries=retries))
            
            if method == "POST":
                logger.info("POST request to %s: %s", endpoint, json.dumps(data, ensure_ascii=False))
                response = session.post(url, json=data, headers=headers, timeout=15)
            else:
                logger.info("GET request to %s", endpoint)
                response = session.get(url, headers=headers, timeout=15)

            response.raise_for_status()
            response_data = response.json()
            logger.info("API response for %s: %s", endpoint, response.text)
            return response_data, None

    except requests.HTTPError as e:
        try:
            # Try to parse JSON error response
            error_data = e.response.json()
            error_message = error_data.get("detail", str(error_data))
            
            # Handle Pydantic validation errors
            if isinstance(error_data, list):
                error_message = "; ".join(
                    f"{err['loc'][-1]}: {err['msg']}" 
                    for err in error_data
                    if isinstance(err, dict)
                )
                
            logger.error("HTTP error for %s: status=%s, message=%s", 
                        endpoint, e.response.status_code, error_message)
            return None, error_message
            
        except ValueError:
            # Non-JSON response
            error_message = e.response.text if e.response else "No response received"
            logger.error("Non-JSON error response: %s", error_message)
            return None, error_message
            
    except requests.Timeout:
        logger.error("Timeout for %s", endpoint)
        return None, "Request timed out. Please try again."
        
    except requests.ConnectionError as e:
        logger.error("Connection error for %s: %s", endpoint, str(e))
        return None, f"Cannot connect to server: {str(e)}"
        
    except Exception as e:
        logger.critical("Unexpected error for %s: %s", endpoint, str(e))
        return None, f"Unexpected error: {str(e)}"

def fetch_stock_data(stock_name: str, days: int = 30) -> tuple[pd.DataFrame, str | None]:
    """Fetch historical and predicted stock data from the database.

    Args:
        stock_name: Stock name ('Reliance' or 'Axis Bank').
        days: Number of days of historical data to fetch.

    Returns:
        tuple: (DataFrame with stock data, error message if any).

    Raises:
        sqlite3.Error: For database connection or query errors.
    """
    logger.debug("Fetching stock data for %s, days=%s", stock_name, days)
    conn = None
    try:
        conn = sqlite3.connect(os.getenv("DATABASE_PATH"), timeout=10)
        cursor = conn.cursor()

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        # Fetch historical data
        cursor.execute(
            """
            SELECT date, close
            FROM stock_data
            WHERE stock_name = ? AND date >= ?
            ORDER BY date ASC
            """,
            (stock_name, start_date.strftime("%Y-%m-%d")),
        )
        stock_data = cursor.fetchall()

        # Fetch predictions
        cursor.execute(
            """
            SELECT target_date, predicted_price
            FROM daily_predictions
            WHERE stock_name = ?
            ORDER BY target_date ASC
            """,
            (stock_name,),
        )
        prediction_data = cursor.fetchall()

        # Process data
        if not stock_data:
            logger.warning("No stock data found for %s", stock_name)
            return pd.DataFrame(), "No stock data available"

        df_stock = pd.DataFrame(stock_data, columns=["Date", "Close"])
        df_stock["Date"] = pd.to_datetime(df_stock["Date"])

        if prediction_data:
            df_pred = pd.DataFrame(prediction_data, columns=["Date", "Predicted"])
            df_pred["Date"] = pd.to_datetime(df_pred["Date"])
            df = df_stock.merge(df_pred, on="Date", how="left")
        else:
            df = df_stock
            df["Predicted"] = pd.NA

        df["Error"] = (df["Close"] - df["Predicted"]).abs().where(df["Predicted"].notna(), pd.NA)
        logger.info("Fetched %s stock data points for %s", len(df), stock_name)
        return df, None

    except sqlite3.Error as e:
        error_message = (
            "Database connection failed: Please try again later."
            if "database is locked" in str(e).lower()
            else f"Database error: {str(e)}"
        )
        logger.error("Database error fetching stock data for %s: %s", stock_name, e)
        return pd.DataFrame(), error_message
    except Exception as e:
        logger.critical("Unexpected error fetching stock data for %s: %s", stock_name, e)
        return pd.DataFrame(), "Unexpected error occurred"
    finally:
        if conn:
            conn.close()


def create_stock_chart(df: pd.DataFrame, stock_name: str) -> dict:
    """Create a Plotly chart for stock prices and predictions.

    Args:
        df: DataFrame with Date, Close, Predicted, and Error columns.
        stock_name: Stock name for chart title.

    Returns:
        dict: Plotly figure configuration.
    """
    logger.debug("Creating stock chart for %s", stock_name)
    fig = {
        "data": [
            {
                "x": df["Date"],
                "y": df["Close"],
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Historical Close",
                "line": {"color": "#2196F3", "width": 2},
                "marker": {"size": 8, "symbol": "circle"},
                "hovertemplate": "Date: %{x|%Y-%m-%d}<br>Close: ₹%{y:.2f}<extra></extra>",
            },
            {
                "x": df["Date"],
                "y": df["Predicted"],
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Predicted Close",
                "line": {"color": "#4CAF50", "width": 2, "dash": "dash"},
                "marker": {"size": 8, "symbol": "diamond"},
                "hovertemplate": "Date: %{x|%Y-%m-%d}<br>Predicted: ₹%{y:.2f}<br>Error: ₹%{customdata:.2f}<extra></extra>",
                "customdata": df["Error"],
            },
        ],
        "layout": {
            "title": f"{stock_name} Stock Price (Last 30 Days)",
            "titlefont": {"color": "white", "size": 20},
            "showlegend": True,
            "legend": {"font": {"color": "white"}},
            "xaxis": {
                "title": "Date",
                "titlefont": {"color": "white"},
                "tickfont": {"color": "white"},
                "gridcolor": "rgba(255, 255, 255, 0.1)",
                "tickangle": 45,
                "tickformat": "%Y-%m-%d",
                "tickmode": "auto",
                "nticks": 10,
            },
            "yaxis": {
                "title": "Price (₹)",
                "titlefont": {"color": "white"},
                "tickfont": {"color": "white"},
                "gridcolor": "rgba(255, 255, 255, 0.1)",
            },
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 40, "r": 40, "t": 60, "b": 80},
            "hovermode": "x unified",
        },
    }
    logger.info("Stock chart created for %s", stock_name)
    return fig


def display_metrics(df: pd.DataFrame, period_days: int, stock_name: str) -> None:
    """Display key metrics for stock performance.

    Args:
        df: DataFrame with stock data.
        period_days: Number of days for performance calculation (7 or 14).
        stock_name: Stock name for display.
    """
    logger.debug("Displaying metrics for %s, period=%s days", stock_name, period_days)
    try:
        end_date = datetime.now().date()
        performance_start_date = end_date - timedelta(days=period_days)
        df_performance = df[df["Date"] >= pd.to_datetime(performance_start_date)]
        mae = df_performance["Error"].mean() if df_performance["Predicted"].notna().any() else None
        mae_display = f"₹{mae:.2f}" if pd.notna(mae) else "N/A"

        metric_cols = st.columns(3)
        with metric_cols[0]:
            latest_close = df["Close"].iloc[-1] if not df["Close"].empty else "N/A"
            st.markdown(
                f"""
                <div class="card card-trade">
                    <div class="card-header">
                        <span class="card-title">LATEST CLOSE PRICE</span>
                    </div>
                    <div class="card-value">₹{latest_close:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_cols[1]:
            if "Predicted" in df.columns and not df["Predicted"].isna().all():
                latest_pred = df["Predicted"].iloc[-1]
                if pd.notna(latest_pred):
                    pred_change = ((latest_pred / latest_close) - 1) * 100 if latest_close != "N/A" else 0
                    change_color = "#4CAF50" if pred_change >= 0 else "#f44336"
                    change_arrow = "↑" if pred_change >= 0 else "↓"
                    st.markdown(
                        f"""
                        <div class="card card-trade">
                            <div class="card-header">
                                <span class="card-title">PREDICTED CLOSE</span>
                            </div>
                            <div class="card-value">₹{latest_pred:,.2f} <span style="font-size: 1rem; color: {change_color};">{change_arrow} {abs(pred_change):.2f}%</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="card card-trade">
                            <div class="card-header">
                                <span class="card-title">PREDICTED CLOSE</span>
                            </div>
                            <div class="card-value">N/A</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                    <div class="card card-trade">
                        <div class="card-header">
                            <span class="card-title">PREDICTED CLOSE</span>
                        </div>
                        <div class="card-value">N/A</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with metric_cols[2]:
            st.markdown(
                f"""
                <div class="card card-performance">
                    <div class="card-header">
                        <span class="card-title">MODEL PERFORMANCE (LAST {period_days} DAYS)</span>
                    </div>
                    <div class="card-value">{mae_display}</div>
                    <div style="font-size: 0.9rem; color: #90a4ae;">Mean Absolute Error</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        logger.info("Metrics displayed for %s", stock_name)
    except Exception as e:
        logger.error("Error displaying metrics for %s: %s", stock_name, e)
        st.error(f"⚠️ Failed to display metrics: {str(e)}")


def display_balance() -> None:
    """Display the user's account balance."""
    logger.debug("Fetching and displaying balance")
    balance_data, balance_error = make_api_request("/balance")
    balance_data = balance_data or {"balance": 100000.00}
    if balance_error:
        logger.error("Failed to fetch balance: %s", balance_error)
        st.error(f"⚠️ Failed to fetch balance: {balance_error}")
    else:
        st.markdown(
            f"""
            <div class="card card-balance">
                <div class="card-header">
                    <span class="card-title">ACCOUNT BALANCE</span>
                </div>
                <div class="card-value">₹{balance_data['balance']:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        logger.info("Balance displayed: ₹%s", balance_data['balance'])


def display_open_trades() -> None:
    """Display all open trades."""
    logger.debug("Fetching and displaying open trades")
    open_trades_data, open_trades_error = make_api_request("/open_trades")
    if open_trades_error:
        logger.error("Failed to fetch open trades: %s", open_trades_error)
        st.error(f"⚠️ Failed to fetch open trades: {open_trades_error}")
    elif open_trades_data and open_trades_data["open_trades"]:
        st.markdown('<div class="trade-history-container">', unsafe_allow_html=True)
        for trade in open_trades_data["open_trades"]:
            trade_type_icon = "⚡" if trade["trade_type"] == "intraday" else "📆"
            action_class = "card-buy" if trade["action"] == "BUY" else "card-sell"
            st.markdown(
                f"""
                <div class="card {action_class}">
                    <div class="card-header">
                        <span class="ticker">{trade["stock_name"]}</span>
                        <span><span class="status-indicator status-open"></span> OPEN</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <div>
                            <div style="font-size: 0.85rem; color: #90a4ae;">Action</div>
                            <div style="font-weight: 600;">{trade["action"]}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.85rem; color: #90a4ae;">Type</div>
                            <div style="font-weight: 600;">{trade_type_icon} {trade["trade_type"]}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.85rem; color: #90a4ae;">Lot Size</div>
                            <div style="font-weight: 600;">{trade["lot_size"]}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.85rem; color: #90a4ae;">Price</div>
                            <div style="font-weight: 600;">₹{trade["entry_price"]:.2f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📭</div>
                <div style="color: #90a4ae;">No open trades at the moment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        logger.info("No open trades to display")


def display_trade_history() -> None:
    """Display the user's trade history."""
    logger.debug("Fetching and displaying trade history")
    history_data, history_error = make_api_request("/trade_history")
    if history_error:
        logger.error("Failed to fetch trade history: %s", history_error)
        st.error(f"⚠️ Failed to fetch trade history: {history_error}")
    elif history_data and history_data["trade_history"]:
        st.markdown('<div class="trade-history-container">', unsafe_allow_html=True)
        trades_to_display = history_data["trade_history"][::-1]  # Newest first
        for trade in trades_to_display:
            profit_loss = trade.get("profit_loss", 0) or 0
            profit_loss_class = "profit" if profit_loss > 0 else "loss" if profit_loss < 0 else ""
            profit_loss_sign = "+" if profit_loss > 0 else ""
            trade_type_icon = "⚡" if trade["trade_type"] == "intraday" else "📆"
            action_class = "card-buy" if trade["action"] == "BUY" else "card-sell"
            entry_time = (
                datetime.strptime(trade.get("entry_time", "N/A"), "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
                if trade.get("entry_time") != "N/A"
                else "N/A"
            )
            exit_time = (
                datetime.strptime(trade.get("exit_time", "N/A"), "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
                if trade.get("exit_time") and trade.get("exit_time") != "N/A"
                else "N/A"
            )
            exit_price = trade.get("exit_price", "N/A")
            exit_price_display = f"₹{exit_price:.2f}" if isinstance(exit_price, (int, float)) else "N/A"
            st.markdown(
                f"""
                <div class="card {action_class}" style="margin-bottom: 0.8rem; padding: 1rem;">
                    <div class="card-header">
                        <span class="ticker">{trade["stock_name"]}</span>
                        <span><span class="status-indicator status-closed"></span> CLOSED</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 0.5rem;">
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">Action</div>
                            <div style="font-weight: 600;">{trade["action"]}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">Type</div>
                            <div style="font-weight: 600;">{trade_type_icon} {trade["trade_type"]}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">Entry</div>
                            <div style="font-weight: 600;">₹{trade["entry_price"]:.2f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">Exit</div>
                            <div style="font-weight: 600;">{exit_price_display}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">Lot Size</div>
                            <div style="font-weight: 600;">{trade["lot_size"]}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">P&L</div>
                            <div class="{profit_loss_class}" style="font-weight: 600;">{profit_loss_sign}₹{abs(profit_loss):.2f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">Entry Time</div>
                            <div style="font-weight: 600;">{entry_time}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #90a4ae;">Exit Time</div>
                            <div style="font-weight: 600;">{exit_time}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <div style="color: #90a4ae;">No trade history available</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        logger.info("No trade history to display")


def display_trade_form() -> None:
    """Display the form for placing a new trade."""
    logger.debug("Displaying trade form")
    st.markdown('<div class="trade-form">', unsafe_allow_html=True)
    with st.form("trade_form"):
        trading_cols = st.columns(2)
        with trading_cols[0]:
            stock_name = st.selectbox("Stock", ["Reliance", "Axis Bank"])
            action = st.selectbox("Action", ["BUY", "SELL"])
        with trading_cols[1]:
            trade_type = st.selectbox("Trade Type", ["intraday", "long-term"])
            lot_size = st.number_input("Lot Size", min_value=1, step=1, value=5)
        price_cols = st.columns(2)
        with price_cols[0]:
            entry_price = st.number_input("Entry Price (₹)", min_value=0.01, step=0.01, value=1446.0)
        with price_cols[1]:
            exit_price = st.number_input(
                "Exit Price (₹) - Only for SELL long-term", min_value=0.0, step=0.01, value=0.0
            )
        password = st.text_input(
            "Trade Password", type="password", placeholder="Contact the admin for password"
        )
        confidence_score = st.slider("Confidence Score (%)", 0.0, 100.0, 96.0, format="%.2f")
        submit = st.form_submit_button("PLACE TRADE")

        if submit:
            logger.debug("Trade form submitted: stock=%s, action=%s", stock_name, action)
            if not password:
                logger.warning("Trade form submitted without password")
                st.error("❌ Please enter the trade password.")
            elif password != TRADE_PASSWORD:
                logger.warning("Invalid trade password")
                st.error("❌ Invalid trade password. Please try again.")
            elif lot_size <= 0:
                logger.warning("Invalid lot size: %s", lot_size)
                st.error("❌ Lot size must be greater than 0.")
            elif entry_price <= 0:
                logger.warning("Invalid entry price: %s", entry_price)
                st.error("❌ Entry price must be greater than 0.")
            elif action == "SELL" and trade_type == "long-term" and exit_price <= 0:
                logger.warning("Missing exit price for long-term SELL")
                st.error("❌ Exit price must be provided for long-term SELL trades.")
            else:
                exit_price_value = exit_price if action == "SELL" and trade_type == "long-term" else 0.0
                trade_data = {
                    "stock_name": stock_name.strip(),
                    "action": action,
                    "entry_price": float(entry_price),
                    "confidence_score": float(confidence_score),
                    "lot_size": int(lot_size),
                    "trade_type": trade_type,
                    "exit_price": float(exit_price_value),
                }
                result, error_message = make_api_request("/trade/", method="POST", data=trade_data)
                if error_message:
                    st.error(f"❌ {error_message}")
                elif result:
                    st.success(f"✅ {result.get('message', 'Trade successful')}")
                else:
                    st.error("❌ Unexpected response from server")
    st.markdown("</div>", unsafe_allow_html=True)


def display_exit_trade_form() -> None:
    """Display the form for exiting an open trade."""
    logger.debug("Displaying exit trade form")
    st.markdown('<div class="trade-form">', unsafe_allow_html=True)
    with st.form("exit_trade_form"):
        exit_cols1 = st.columns(2)
        with exit_cols1[0]:
            stock_name = st.selectbox("Stock", ["Reliance", "Axis Bank"], key="exit_stock")
        with exit_cols1[1]:
            trade_type = st.selectbox("Trade Type", ["intraday", "long-term"], key="exit_type")
        exit_cols2 = st.columns(2)
        with exit_cols2[0]:
            exit_price = st.number_input("Exit Price (₹)", min_value=0.0, step=0.01, value=1000.0, key="exit_price")
        with exit_cols2[1]:
            lot_size = st.number_input("Lot Size", min_value=1, step=1, value=1, key="exit_lot")
        password = st.text_input(
            "Trade Password", type="password", key="exit_password", placeholder="Contact the admin for password"
        )
        submit = st.form_submit_button("EXIT TRADE")

        if submit:
            logger.debug("Exit trade form submitted: stock=%s, trade_type=%s", stock_name, trade_type)
            if not password:
                logger.warning("Exit trade form submitted without password")
                st.error("❌ Please enter the trade password.")
            elif password != TRADE_PASSWORD:
                logger.warning("Invalid trade password")
                st.error("❌ Invalid trade password. Please try again.")
            else:
                exit_data = {
                    "stock_name": stock_name.strip(),
                    "exit_price": float(exit_price),
                    "trade_type": trade_type,
                    "lot_size": int(lot_size),
                }
                result, error_message = make_api_request("/exit_trade/", method="POST", data=exit_data)
                if error_message:
                    st.error(f"❌ {error_message}")
                elif result:
                    st.success(f"✅ {result.get('message', 'Trade exited successfully')}")
                else:
                    st.error("❌ Unexpected response from server")
    st.markdown("</div>", unsafe_allow_html=True)


def display_chatbot() -> None:
    """Display the chatbot interface and handle user queries."""
    logger.debug("Displaying chatbot interface")
    st.markdown("<h3>Assistant</h3>", unsafe_allow_html=True)
    chat_container_html = '<div class="chat-container">'
    if not st.session_state.chat_history:
        chat_container_html += """
        <div class="chat-message-bot">
           Hey👋, Welcome to your Stock Trading Assistant!
           Ask me anything about your account balance, trade history, or price predictions for Reliance and Axis.
        </div>
        """
    else:
        for chat in st.session_state.chat_history:
            chat_container_html += f'<div class="chat-message-user">{chat["user"]}</div>'
            chat_container_html += f'<div class="chat-message-bot">{chat["bot"]}</div>'
    chat_container_html += "</div>"
    st.markdown(chat_container_html, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.text_input(
                "Chat input",
                placeholder="Ask about the chart or trades...",
                key="chat_input",
                label_visibility="collapsed",
            )
        with col2:
            submit = st.form_submit_button("Send")
        if submit and user_query:
            logger.debug("Chat query submitted: %s", user_query)
            chat_data = {"query": user_query}
            chat_response, chat_error = make_api_request("/chat", method="POST", data=chat_data)
            if chat_response and "response" in chat_response:
                st.session_state.chat_history.append({"user": user_query, "bot": chat_response["response"]})
                logger.info("Chat response added to history")
                st.rerun()
            elif chat_error:
                logger.error("Chat error: %s", chat_error)
                st.error(f"❌ Chat error: {chat_error}")
            else:
                logger.error("Chat error: Unable to process request")
                st.error("❌ Chat error: Unable to process request.")


def main():
    """Run the XAI TraderX Streamlit application."""
    logger.info("Starting XAI TraderX application")
    try:
        configure_page()
        init_session_state()

        # Header
        st.markdown('<h1 style="text-align: center; margin-bottom: 1rem;">XAI TraderX</h1>', unsafe_allow_html=True)

        # Main layout
        col1, col2 = st.columns([5, 3])

        with col1:
            # Stock Analysis Section
            st.markdown('<div class="section-header">Stock Analysis</div>', unsafe_allow_html=True)
            selected_stock = st.selectbox("Select Stock for Analysis", ["Reliance", "Axis Bank"], key="chart_stock")
            df, stock_error = fetch_stock_data(selected_stock)
            if stock_error:
                logger.error("Failed to fetch stock data: %s", stock_error)
                st.error(f"⚠️ {stock_error}")
            elif not df.empty:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_stock_chart(df, selected_stock)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
                st.markdown("</div>", unsafe_allow_html=True)

                # Model performance
                performance_period = st.selectbox(
                    "Select Performance Period", ["7 Days", "14 Days"], key="performance_period"
                )
                period_days = 7 if performance_period == "7 Days" else 14
                st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
                display_metrics(df, period_days, selected_stock)

            # Trading Section
            st.markdown('<div class="section-header">Trading</div>', unsafe_allow_html=True)
            display_balance()

            # Trading tabs
            tabs = st.tabs(["📊 Open Trades", "📜 Trade History", "🛒 Place Trade", "🚪 Exit Trade"])
            with tabs[0]:
                display_open_trades()
            with tabs[1]:
                display_trade_history()
            with tabs[2]:
                display_trade_form()
            with tabs[3]:
                display_exit_trade_form()

        with col2:
            display_chatbot()

        logger.info("Application rendered successfully")
    except Exception as e:
        logger.critical("Application error: %s", e)
        st.error(f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    main()