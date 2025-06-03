"""Stock prediction and trade information pipeline.

This module implements a Retrieval-Augmented Generation (RAG) pipeline to predict stock prices
for Reliance and Axis Bank and handle user queries about trades, profits, losses, and balance.
It supports natural language queries with typo correction, date parsing, and enhanced profit/loss analysis.
"""

# Standard library imports
import logging
import re
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import sqlite3
from typing import List, Dict, Optional

# Third-party imports
import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
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
        RotatingFileHandler("pipeline.log", maxBytes=1000000, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Allowed stock names
ALLOWED_STOCKS = ["Reliance", "Axis Bank"]


def load_config(config_path: str = "rag_pipeline/config/config.yaml") -> Dict:
    """Load the configuration file for the pipeline.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is missing.
        yaml.YAMLError: If the config file has invalid YAML.
    """
    try:
        logger.debug("Loading config from %s", config_path)
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Config loaded successfully")
            return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in config file: %s", e)
        raise


class RAGPipeline:
    """Pipeline for stock price predictions and trade-related queries.

    Attributes:
        config: Configuration dictionary for embeddings, LLM, and database.
        embedding_model: HuggingFace model for generating text embeddings.
        collection: Chroma collection for vector storage.
        model: Loaded language model for inference.
        tokenizer: Tokenizer for the language model.
        vector_store: Chroma vector store for similarity search.
        last_queried_date: Last date queried by the user for tracking.
    """

    def __init__(self, config: Dict):
        """Initialize the RAG pipeline with configuration.

        Args:
            config: Dictionary containing settings for embeddings, LLM, database.

        Raises:
            ValueError: If config is missing required keys.
            RuntimeError: If pipeline initialization fails.
        """
        try:
            logger.debug("Initializing RAGPipeline")
            required_keys = ["embedding", "chroma", "llm", "database"]
            if not all(key in config for key in required_keys):
                logger.error("Missing required config keys: %s", required_keys)
                raise ValueError("Config missing required keys")

            self.config = config
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=config["embedding"]["model_name"]
            )
            self.collection = init_chroma(
                config["chroma"]["persist_directory"],
                config["chroma"]["collection_name"],
            )
            self.model, self.tokenizer = load_llm(
                config["llm"]["model_name"],
                config["llm"]["device"],
                config["llm"].get("api_key", ""),
            )
            self.vector_store = Chroma(
                collection_name=config["chroma"]["collection_name"],
                embedding_function=self.embedding_model,
                persist_directory=config["chroma"]["persist_directory"],
            )
            self.last_queried_date = None
            logger.info("RAGPipeline initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize RAGPipeline: %s", e)
            raise RuntimeError("RAGPipeline initialization failed")

    def update_pipeline(self) -> None:
        """Update the pipeline with new data.

        Fetches recent trading data, preprocesses it, generates embeddings, and updates the Chroma vector store.

        Raises:
            RuntimeError: If data fetching or updating fails.
        """
        try:
            logger.info("Updating pipeline with new data")
            data = fetch_all_data(self.config["database"]["path"], days_back=30)
            text_chunks = preprocess_data(data)
            embeddings, text_chunks = generate_embeddings(
                text_chunks, self.config["embedding"]["model_name"]
            )
            update_chroma(self.collection, embeddings, text_chunks)
            logger.info("Pipeline updated successfully")
        except Exception as e:
            logger.error("Pipeline update failed: %s", e)
            raise RuntimeError("Pipeline update failed")

    def get_last_trading_days(self, num_days: int) -> List[str]:
        """Get the last N trading days, skipping weekends.

        Args:
            num_days: Number of trading days to fetch.

        Returns:
            List[str]: List of dates in YYYY-MM-DD format.

        Raises:
            ValueError: If num_days is not positive.
        """
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
            logger.error("Failed to fetch trading days: %s", e)
            raise

    def get_prediction_accuracy(self, stock_name: str, specific_date: Optional[str] = None) -> Dict:
        """Calculate prediction accuracy for a stock over a period or specific date.

        Args:
            stock_name: Name of the stock (e.g., 'Reliance', 'Axis Bank').
            specific_date: Optional date in YYYY-MM-DD format to check accuracy for a single day.

        Returns:
            Dict: Dictionary with accuracy metrics (e.g., mean absolute error, accuracy percentage).

        Raises:
            RuntimeError: If database query fails.
        """
        try:
            logger.debug("Calculating prediction accuracy for %s on %s", stock_name, specific_date or "recent period")
            conn = sqlite3.connect(self.config["database"]["path"], timeout=10)
            cursor = conn.cursor()
            query = """
                SELECT predicted_price, actual_price
                FROM daily_predictions
                WHERE stock_name = ? AND actual_price IS NOT NULL
            """
            params = [stock_name]
            if specific_date:
                query += " AND target_date = ?"
                params.append(specific_date)
            else:
                query += " ORDER BY target_date DESC LIMIT 30"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            if not rows:
                logger.warning("No valid predictions with actual prices for %s", stock_name)
                return {"mean_absolute_error": None, "accuracy_percentage": None, "message": "No data available"}
            
            errors = [abs(row[0] - row[1]) for row in rows]
            mean_absolute_error = sum(errors) / len(errors)
            within_5_percent = sum(1 for row in rows if abs(row[0] - row[1]) / row[1] <= 0.05) / len(rows) * 100
            logger.info("Prediction accuracy for %s: MAE=%.2f, Accuracy=%.1f%%", stock_name, mean_absolute_error, within_5_percent)
            return {
                "mean_absolute_error": round(mean_absolute_error, 2),
                "accuracy_percentage": round(within_5_percent, 1),
                "message": f"Predictions within 5% of actual price: {within_5_percent:.1f}%"
            }
        except sqlite3.Error as e:
            logger.error("Database error calculating prediction accuracy: %s", e)
            raise RuntimeError(f"Failed to calculate prediction accuracy: {str(e)}")
        finally:
            conn.close()

    def get_most_profitable_day(self, trades: List[Dict], stock_name: Optional[str] = None) -> Dict:
        """Identify the most profitable trading day based on trade data.

        Args:
            trades: List of trade dictionaries.
            stock_name: Optional stock name to filter trades.

        Returns:
            Dict: Details of the most profitable day (date, total profit, trades).
        """
        try:
            logger.debug("Finding most profitable day for %s", stock_name or "all stocks")
            if not trades:
                return {"date": None, "total_profit": 0, "trades": [], "message": "No trades found"}
            
            filtered_trades = [t for t in trades if not stock_name or stock_name.lower() in t["stock"].lower()]
            profit_by_date = {}
            for trade in filtered_trades:
                if trade["profit_loss"] > 0:
                    date = trade["entry_date"]
                    profit_by_date[date] = profit_by_date.get(date, 0) + trade["profit_loss"]
            
            if not profit_by_date:
                return {"date": None, "total_profit": 0, "trades": [], "message": "No profitable trades found"}
            
            most_profitable_date = max(profit_by_date, key=profit_by_date.get)
            relevant_trades = [t for t in filtered_trades if t["entry_date"] == most_profitable_date and t["profit_loss"] > 0]
            logger.info("Most profitable day: %s with ₹%.2f", most_profitable_date, profit_by_date[most_profitable_date])
            return {
                "date": most_profitable_date,
                "total_profit": round(profit_by_date[most_profitable_date], 2),
                "trades": relevant_trades,
                "message": f"Most profitable day was {most_profitable_date} with ₹{profit_by_date[most_profitable_date]:.2f} profit"
            }
        except Exception as e:
            logger.error("Failed to find most profitable day: %s", e)
            return {"date": None, "total_profit": 0, "trades": [], "message": f"Error: {str(e)}"}

    def get_most_profitable_stock(self, trades: List[Dict], days_back: int) -> Dict:
        """Identify the most profitable stock over a period.

        Args:
            trades: List of trade dictionaries.
            days_back: Number of days to consider.

        Returns:
            Dict: Details of the most profitable stock (name, total profit, trades).
        """
        try:
            logger.debug("Finding most profitable stock over %d days", days_back)
            trading_days = self.get_last_trading_days(days_back)
            filtered_trades = [t for t in trades if t["entry_date"] in trading_days]
            profit_by_stock = {}
            for trade in filtered_trades:
                if trade["profit_loss"] > 0:
                    stock = trade["stock"]
                    profit_by_stock[stock] = profit_by_stock.get(stock, 0) + trade["profit_loss"]
            
            if not profit_by_stock:
                return {"stock": None, "total_profit": 0, "trades": [], "message": "No profitable trades found"}
            
            most_profitable_stock = max(profit_by_stock, key=profit_by_stock.get)
            relevant_trades = [t for t in filtered_trades if t["stock"] == most_profitable_stock and t["profit_loss"] > 0]
            logger.info("Most profitable stock: %s with ₹%.2f", most_profitable_stock, profit_by_stock[most_profitable_stock])
            return {
                "stock": most_profitable_stock,
                "total_profit": round(profit_by_stock[most_profitable_stock], 2),
                "trades": relevant_trades,
                "message": f"Most profitable stock was {most_profitable_stock} with ₹{profit_by_stock[most_profitable_stock]:.2f} profit"
            }
        except Exception as e:
            logger.error("Failed to find most profitable stock: %s", e)
            return {"stock": None, "total_profit": 0, "trades": [], "message": f"Error: {str(e)}"}

    def format_trade_response(
        self,
        trades: List[Dict],
        query_type: str,
        specific_date: Optional[str] = None,
        stock_name: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> str:
        """Format trade data into a user-friendly response.

        Args:
            trades: List of trade dictionaries.
            query_type: Type of query ('last_one', 'intraday', 'profit', 'loss', 'last_x_days', etc.).
            specific_date: Optional date to filter trades (YYYY-MM-DD).
            stock_name: Optional stock name to filter trades.
            days_back: Optional number of days for 'last_x_days' query.

        Returns:
            str: Formatted response summarizing trade details.
        """
        try:
            logger.debug("Formatting trade response: query_type=%s, trades=%d", query_type, len(trades))
            if not trades:
                date_str = f" on {specific_date}" if specific_date else ""
                stock_str = f" for {stock_name}" if stock_name else ""
                if query_type == "last_x_days" and days_back is not None:
                    return f"No trades found in the last {days_back} days{stock_str}. Want to check recent trades or predictions?"
                return f"No trades found{stock_str}{date_str}. Try a different date, stock, or check predictions."

            total_profit = sum(t["profit_loss"] for t in trades if t["profit_loss"] is not None and t["profit_loss"] > 0)
            total_loss = sum(t["profit_loss"] for t in trades if t["profit_loss"] is not None and t["profit_loss"] < 0)
            stocks_traded = list(set(t["stock"] for t in trades))
            date_str = f" on {specific_date}" if specific_date else ""
            stock_str = f" for {stock_name}" if stock_name else ""

            if query_type == "last_x_days":
                response = f"In the last {days_back} days, you made {len(trades)} trade{'s' if len(trades) > 1 else ''} on {', '.join(stocks_traded)}, earning ₹{total_profit:.2f} in profits"
                response += f" and ₹{-total_loss:.2f} in losses." if total_loss < 0 else " with no losses."
                response += "\n\nDetails:\n"
                for trade in trades:
                    response += f"- {trade['stock']}: {trade['action']} ({trade['trade_type_label']}) on {trade['entry_date']}, {'Profit' if trade['profit_loss'] >= 0 else 'Loss'}: ₹{abs(trade['profit_loss']):.2f}, Lot Size: {trade['lot_size']}\n"
                response += "\nWant a chart of your profits or more details?"
                logger.info("Formatted last_x_days trade response")
                return response

            if query_type == "last_one":
                trade = trades[-1]
                profit_loss_str = (
                    f"profit of ₹{trade['profit_loss']:.2f}"
                    if trade["profit_loss"] >= 0
                    else f"loss of ₹{-trade['profit_loss']:.2f}"
                )
                response = (
                    f"Your most recent trade{stock_str}{date_str} was with {trade['stock']}. "
                    f"You performed a {trade['trade_type_label']} ({trade['trade_type']}) on {trade['entry_date']}, "
                    f"earning a {profit_loss_str} with a lot size of {trade['lot_size']}. "
                    f"Trade started at {trade['entry_time']} and ended at {trade['exit_time']} (Status: {trade['status']})."
                )
                response += "\n\nWant to see more trades or predictions?"
                logger.info("Formatted last trade response")
                return response

            if query_type == "intraday":
                response = (
                    f"Your intraday trades{stock_str}{date_str}: "
                    f"{len(trades)} trade{'s' if len(trades) > 1 else ''} involving "
                    f"{len(stocks_traded)} stock{'s' if len(stocks_traded) > 1 else ''}: {', '.join(stocks_traded)}. "
                )
                if total_profit > 0:
                    response += f"You earned ₹{total_profit:.2f} in profits"
                    response += (
                        f", but faced ₹{-total_loss:.2f} in losses."
                        if total_loss < 0
                        else ", with no losses."
                    )
                else:
                    response += "No profits this time."
                response += "\n\nHere's what happened:\n"
                for trade in trades:
                    profit_loss_str = (
                        f"Profit of ₹{trade['profit_loss']:.2f}"
                        if trade["profit_loss"] >= 0
                        else f"Loss of ₹{-trade['profit_loss']:.2f}"
                    )
                    response += (
                        f"- {trade['stock']}: {trade['trade_type_label']} ({trade['trade_type']}), {profit_loss_str}, "
                        f"Lot Size: {trade['lot_size']}, Entered at {trade['entry_time']}, Exited at {trade['exit_time']}\n"
                    )
                response += "\nWant a profit chart or predictions?"
                logger.info("Formatted intraday trade response")
                return response

            if query_type == "profit":
                profitable_trades = [
                    t for t in trades if t["profit_loss"] is not None and t["profit_loss"] > 0
                ]
                if not profitable_trades:
                    logger.info("No profitable trades found%s%s", stock_str, date_str)
                    return f"No profitable trades found{stock_str}{date_str}. Want to explore other dates, stocks, or predictions?"
                response = (
                    f"Your profits{stock_str}{date_str}: "
                    f"₹{total_profit:.2f} from {len(profitable_trades)} winning trade{'s' if len(profitable_trades) > 1 else ''} "
                    f"across {len(stocks_traded)} stock{'s' if len(stocks_traded) > 1 else ''}: {', '.join(stocks_traded)}. "
                )
                response += "\n\nDetails:\n"
                for trade in profitable_trades:
                    response += (
                        f"- {trade['stock']}: {trade['trade_type_label']} on {trade['entry_date']}, "
                        f"Profit: ₹{trade['profit_loss']:.2f}, Lot Size: {trade['lot_size']}\n"
                    )
                response += "\nWant a profit chart or more trade details?"
                logger.info("Formatted profit trade response")
                return response

            if query_type == "loss":
                loss_trades = [
                    t for t in trades if t["profit_loss"] is not None and t["profit_loss"] < 0
                ]
                if not loss_trades:
                    logger.info("No losing trades found%s%s", stock_str, date_str)
                    return f"No losing trades found{stock_str}{date_str}. Want to check profitable trades or predictions?"
                response = (
                    f"Your losing trades{stock_str}{date_str}: ₹{-total_loss:.2f} from "
                    f"{len(loss_trades)} trade{'s' if len(loss_trades) > 1 else ''} on {', '.join(stocks_traded)}. "
                )
                response += "\n\nDetails:\n"
                for trade in loss_trades:
                    response += (
                        f"- {trade['stock']}: {trade['trade_type_label']} on {trade['entry_date']}, "
                        f"Loss: ₹{-trade['profit_loss']:.2f}, Lot Size: {trade['lot_size']}\n"
                    )
                response += "\nWant to analyze these losses or see predictions?"
                logger.info("Formatted loss trade response")
                return response

            if query_type == "trade_dates":
                trade_dates = sorted(set(t["entry_date"] for t in trades))
                response = f"Your trade dates{stock_str}:\n"
                for date in trade_dates:
                    response += f"- {date}\n"
                response += "\nWant to see trades for any of these days or a profit chart?"
                logger.info("Formatted trade_dates response")
                return response

            if query_type == "trade_count":
                response = (
                    f"You made {len(trades)} trade{'s' if len(trades) != 1 else ''}{stock_str}{date_str}. "
                )
                response += "\nWant to see trade details or a summary chart?"
                logger.info("Formatted trade_count response")
                return response

            if query_type == "most_profitable_day":
                most_profitable = self.get_most_profitable_day(trades, stock_name)
                if not most_profitable["date"]:
                    return most_profitable["message"]
                response = (
                    f"Your most profitable day{stock_str} was {most_profitable['date']} with ₹{most_profitable['total_profit']:.2f} profit."
                    f"\n\nDetails:\n"
                )
                for trade in most_profitable["trades"]:
                    response += (
                        f"- {trade['stock']}: {trade['trade_type_label']} on {trade['entry_date']}, "
                        f"Profit: ₹{trade['profit_loss']:.2f}, Lot Size: {trade['lot_size']}\n"
                    )
                response += "\nWant a chart of daily profits or more details?"
                logger.info("Formatted most profitable day response")
                return response

            if query_type == "most_profitable_stock":
                most_profitable = self.get_most_profitable_stock(trades, days_back or 30)
                if not most_profitable["stock"]:
                    return most_profitable["message"]
                response = (
                    f"Your most profitable stock over the last {days_back or 30} days was {most_profitable['stock']} "
                    f"with ₹{most_profitable['total_profit']:.2f} profit."
                    f"\n\nDetails:\n"
                )
                for trade in most_profitable["trades"]:
                    response += (
                        f"- {trade['stock']}: {trade['trade_type_label']} on {trade['entry_date']}, "
                        f"Profit: ₹{trade['profit_loss']:.2f}, Lot Size: {trade['lot_size']}\n"
                    )
                response += "\nWant a chart of stock performance or more details?"
                logger.info("Formatted most profitable stock response")
                return response

            trades_to_show = trades[-4:] if len(trades) >= 4 else trades
            response = (
                f"Your recent trades{stock_str}{date_str}: "
                f"{len(trades)} trade{'s' if len(trades) > 1 else ''} across "
                f"{len(stocks_traded)} stock{'s' if len(stocks_traded) > 1 else ''}: {', '.join(stocks_traded)}. "
            )
            if total_profit > 0:
                response += f"You earned ₹{total_profit:.2f} in profits"
                response += (
                    f", with ₹{-total_loss:.2f} in losses." if total_loss < 0 else ", and no losses."
                )
            else:
                response += "No profits this time."
            response += f"\n\nHere are your {'latest ' + str(len(trades_to_show)) + ' ' if len(trades) > 4 else ''}trades:\n"
            for trade in trades_to_show:
                profit_loss_str = (
                    f"Profit of ₹{trade['profit_loss']:.2f}"
                    if trade["profit_loss"] >= 0
                    else f"Loss of ₹{-trade['profit_loss']:.2f}"
                )
                response += (
                    f"- {trade['stock']}: {trade['trade_type_label']} on {trade['entry_date']}, {profit_loss_str}, "
                    f"Lot Size: {trade['lot_size']}, Status: {trade['status']}, Entered at {trade['entry_time']}, "
                    f"Exited at {trade['exit_time']}\n"
                )
                response += "\nWant a chart of your trades or predictions?"
            logger.info("Formatted general trade response")
            return response

        except Exception as e:
            logger.error("Failed to format trade response: %s", e)
            return f"Error processing trades: {str(e)}. Try a different query or check predictions."

    def generate_profit_chart(self, trades: List[Dict], days_back: int, stock_name: Optional[str] = None) -> str:
        """Generate a Chart.js configuration for visualizing profit/loss over time.

        Args:
            trades: List of trade dictionaries.
            days_back: Number of days to include in the chart.
            stock_name: Optional stock name to filter trades.

        Returns:
            str: Chart.js configuration as a JSON string.
        """
        try:
            logger.debug("Generating profit chart for %d days, stock=%s", days_back, stock_name or "all")
            trading_days = self.get_last_trading_days(days_back)
            profit_by_date = {date: 0 for date in trading_days}
            filtered_trades = [t for t in trades if t["entry_date"] in trading_days and (not stock_name or stock_name.lower() in t["stock"].lower())]
            
            for trade in filtered_trades:
                profit_by_date[trade["entry_date"]] += trade["profit_loss"]
            
            dates = sorted(profit_by_date.keys())
            profits = [profit_by_date[date] for date in dates]
            
            chart_config = {
                "type": "bar",
                "data": {
                    "labels": dates,
                    "datasets": [{
                        "label": f"Profit/Loss ({stock_name or 'All Stocks'})",
                        "data": profits,
                        "backgroundColor": ["#4CAF50" if p >= 0 else "#F44336" for p in profits],
                        "borderColor": "#333333",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {"display": True, "text": "Profit/Loss (₹)"}
                        },
                        "x": {
                            "title": {"display": True, "text": "Date"}
                        }
                    },
                    "plugins": {
                        "title": {"display": True, "text": f"Profit/Loss Over Last {days_back} Days"}
                    }
                }
            }
            logger.info("Generated profit chart for %d days", days_back)
            return f"```chartjs\n{chart_config}\n```"
        except Exception as e:
            logger.error("Failed to generate profit chart: %s", e)
            return f"Error generating chart: {str(e)}"

    def suggest_profit_strategy(
        self,
        target_profit: float,
        stock_price: float,
        balance: float,
        trades: List[Dict],
        days_left_in_month: int,
    ) -> str:
        """Suggest a trading strategy to achieve a profit goal.

        Args:
            target_profit: Target profit amount in INR.
            stock_price: Current price of the stock (unused, kept for compatibility).
            balance: Current user balance in INR.
            trades: List of past trade dictionaries.
            days_left_in_month: Number of trading days left in the month.

        Returns:
            str: Recommended trading strategy.
        """
        try:
            logger.debug("Suggesting profit strategy for target_profit=₹%.2f", target_profit)
            avg_profit_per_trade = 0
            trade_count = len(trades)
            if trade_count > 0:
                total_profit = sum(
                    t["profit_loss"]
                    for t in trades
                    if t["profit_loss"] is not None and t["profit_loss"] > 0
                )
                profitable_trades = len(
                    [
                        t
                        for t in trades
                        if t["profit_loss"] is not None and t["profit_loss"] > 0
                    ]
                )
                avg_profit_per_trade = total_profit / profitable_trades if profitable_trades > 0 else 0

            remaining_profit_needed = target_profit
            trades_needed = (
                int(remaining_profit_needed / avg_profit_per_trade) + 1
                if avg_profit_per_trade > 0
                else "several"
            )
            risk_percentage = min(0.02, remaining_profit_needed / balance) if balance > 0 else 0.02
            capital_per_trade = balance * risk_percentage

            most_profitable_stock = self.get_most_profitable_stock(trades, 30)
            recommended_stock = most_profitable_stock["stock"] or "Reliance"

            response = (
                f"To reach your goal of ₹{target_profit:,.2f} profit this month, "
                f"you need to make ₹{remaining_profit_needed:,.2f} more. "
            )
            if trade_count > 0 and avg_profit_per_trade > 0:
                response += (
                    f"Based on your past trades, you've averaged ₹{avg_profit_per_trade:.2f} profit per winning trade. "
                    f"You might need around {trades_needed} successful trades."
                )
            else:
                response += "You haven't had profitable trades recently, so let's plan carefully."
            response += (
                f"\nWith your balance of ₹{balance:,.2f}, consider risking about ₹{capital_per_trade:,.2f} per trade "
                f"(roughly {risk_percentage*100:.1f}% of your capital) to stay safe. "
                f"Focus on {recommended_stock}, which has been your most profitable stock recently. "
                f"Check predictions for the next few days to find good entry points. "
                f"With {days_left_in_month} trading days left, aim for 1-2 trades per week with clear stop-losses. "
                f"Want to see predictions for {recommended_stock} or a profit chart?"
            )
            logger.info("Generated profit strategy for target=₹%.2f", target_profit)
            return response
        except Exception as e:
            logger.error("Failed to suggest profit strategy: %s", e)
            return f"Error creating strategy: {str(e)}. Try checking your balance or trades."

    def get_latest_prediction_date(self) -> Optional[str]:
        """Get the latest date for stock predictions.

        Returns:
            Optional[str]: Latest prediction date in YYYY-MM-DD format, or None if no valid predictions.

        Raises:
            RuntimeError: If database query fails.
        """
        logger.debug("Fetching latest prediction date")
        conn = None
        try:
            conn = sqlite3.connect(self.config["database"]["path"], timeout=10)
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
            logger.error("Database error fetching latest prediction date: %s", e)
            raise RuntimeError(f"Failed to fetch latest prediction date: {str(e)}")
        finally:
            if conn:
                conn.close()

    def _process_trades(self, trade_data: List[tuple]) -> List[Dict]:
        """Process trade data into a structured format.

        Args:
            trade_data: List of trade tuples from the database.

        Returns:
            List[Dict]: List of processed trade dictionaries.
        """
        trades = []
        for trade in trade_data:
            trade_id, stock_name_db, action, entry_time, entry_price, confidence_score, exit_time, exit_price, profit_loss, status, lot_size, trade_type, is_short = trade
            entry_date = entry_time.split(" ")[0]
            if status == "CLOSED":
                profit_loss_value = float(profit_loss) if profit_loss is not None else 0.0
                trade_type_label = "Short" if is_short else "Buy"
                trades.append({
                    "stock": stock_name_db,
                    "action": action,
                    "entry_date": entry_date,
                    "profit_loss": profit_loss_value,
                    "lot_size": int(lot_size) if lot_size is not None else 1,
                    "entry_time": entry_time,
                    "exit_time": exit_time or "None",
                    "status": status,
                    "trade_type": trade_type,
                    "trade_type_label": trade_type_label,
                })
        return trades

    def query(self, user_query: str) -> str:
        """Process a user query and generate a response.

        Args:
            user_query: User input string (e.g., "show trades for Reliance").

        Returns:
            str: Response to the query (e.g., trade details, predictions, charts).

        Raises:
            ValueError: If query processing fails due to invalid input.
            RuntimeError: If data fetching or LLM inference fails.
        """
        try:
            logger.debug("Processing query: %s", user_query)
            days_back = 30
            specific_date = None
            user_query_lower = user_query.lower().strip()

            # Extended typo corrections
            typo_corrections = {
                "pasta": "past",
                "predction": "prediction",
                "predicition": "prediction",
                "prediciin": "prediction",
                "forcast": "forecast",
                "tarde": "trade",
                "tad": "trade",
                "otmmwor": "tomorrow",
                "tommorow": "tomorrow",
                "yestr": "yesterday",
                "rleicen": "reliance",
                "releicn": "reliance",
                "azisa": "axis",
                "axisa": "axis",
                "clso eprioce": "close price",
                "clso": "close",
                "prioce": "price",
                "das": "days",
                "n": "and",
                "monlty": "monthly",
                "profita": "profit",
                "lsos": "loss",
                "prfy": "profit",
                "godo": "good",
                "teh": "the",
                "anwre": "answer",
                "predcitioed": "predicted"
            }
            for typo, correct in typo_corrections.items():
                user_query_lower = user_query_lower.replace(typo, correct)

            # Handle greeting or vague queries
            if any(phrase in user_query_lower for phrase in ["hey", "hi", "hello", "what can you do"]):
                logger.info("Greeting query detected")
                return (
                    "Hey there! I'm your trading assistant for Reliance and Axis Bank. I can predict stock prices, "
                    "show trades, profits, losses, or your balance. I can also show charts for profits or predictions. "
                    "Try 'What's Reliance's price tomorrow?', 'Show my trades', or 'Which stock was most profitable?'"
                )

            # Parse date-related queries
            if "today" in user_query_lower:
                specific_date = datetime.now().strftime("%Y-%m-%d")
                self.last_queried_date = specific_date
                logger.debug("Set specific date to today: %s", specific_date)
            elif "yesterday" in user_query_lower:
                yesterday = datetime.now() - timedelta(days=1)
                while yesterday.weekday() >= 5:  # Skip weekends
                    yesterday -= timedelta(days=1)
                specific_date = yesterday.strftime("%Y-%m-%d")
                self.last_queried_date = specific_date
                logger.debug("Set specific date to yesterday: %s", specific_date)
            elif "tomorrow" in user_query_lower:
                tomorrow = datetime.now() + timedelta(days=1)
                while tomorrow.weekday() >= 5:  # Skip weekends
                    tomorrow += timedelta(days=1)
                specific_date = tomorrow.strftime("%Y-%m-%d")
                self.last_queried_date = specific_date
                logger.debug("Set specific date to tomorrow: %s", specific_date)
            elif "that day" in user_query_lower or "that i asked" in user_query_lower:
                specific_date = self.last_queried_date or datetime.now().strftime("%Y-%m-%d")
                logger.debug("Using previous date: %s", specific_date)
            elif "last week" in user_query_lower:
                days_back = 7
                logger.debug("Set days_back to last week: %s", days_back)
            elif "this month" in user_query_lower or "monthly" in user_query_lower:
                days_back = 30
                logger.debug("Set days_back to this month: %s", days_back)

            date_match = re.search(r"(\d{4}/\d{2}/\d{2})|(\d{1,2}/\d{1,2}/\d{4})", user_query_lower)
            if date_match:
                try:
                    date_str = date_match.group(1) or date_match.group(2)
                    date_format = "%Y/%m/%d" if len(date_str.split("/")[0]) == 4 else "%d/%m/%Y"
                    specific_date = datetime.strptime(date_str, date_format).strftime("%Y-%m-%d")
                    self.last_queried_date = specific_date
                    logger.debug("Parsed date: %s", specific_date)
                except ValueError:
                    logger.warning("Invalid date format in query")
                    return "Invalid date format. Use yyyy/mm/dd or dd/mm/yyyy. Try again or ask for recent trades."

            # Handle weekday queries
            days_of_week = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6
            }
            for day_name, day_offset in days_of_week.items():
                if day_name in user_query_lower:
                    today = datetime.now()
                    current_day = today.weekday()
                    days_until_day = (day_offset - current_day) % 7 or 7
                    specific_date = (today - timedelta(days=days_until_day)).strftime("%Y-%m-%d")
                    self.last_queried_date = specific_date
                    logger.debug("Set specific day to %s for %s", specific_date, day_name)
                    break

            # Handle "last N days" queries
            days_match = re.search(r"last (\d+) days?", user_query_lower)
            if days_match:
                days_back = int(days_match.group(1))
                logger.debug("Set days_back to: %s", days_back)

            # Fetch data from database
            try:
                data = fetch_all_data(
                    self.config["database"]["path"],
                    days_back=days_back,
                    specific_date=specific_date,
                )
            except Exception as e:
                logger.error("Failed to fetch data: %s", str(e))
                return "Failed to fetch data. Try again later or check recent predictions."

            if not (data["daily_predictions"] or data["trades"] or data["user_balance"]):
                logger.warning("No data found")
                return f"No data found for {specific_date or 'this period'}. Try another date, stock, or check your balance."

            # Identify stock
            stock_name = None
            stock_keywords = {
                "reliance": "Reliance",
                "axis": "Axis Bank",
            }
            for keyword, actual_name in stock_keywords.items():
                if keyword in user_query_lower:
                    stock_name = actual_name
                    logger.debug("Identified stock: %s", stock_name)
                    break

            # Handle balance query
            if "balance" in user_query_lower:
                balance = data["user_balance"][0][1] if data["user_balance"] else 0
                logger.info("Balance retrieved: ₹%s", balance)
                return f"Your balance is ₹{float(balance):,.2f}. Want to make a trade or see predictions?"

            # Handle profit goal query
            profit_goal_match = re.search(r"(?:make|earn|want)\s*(\d+\.?\d*)\s*(?:k|thousand)?\s*profit", user_query_lower)
            if profit_goal_match:
                target_profit = float(profit_goal_match.group(1)) * 1000 if "k" in user_query_lower or "thousand" in user_query_lower else float(profit_goal_match.group(1))
                balance = data["user_balance"][0][1] if data["user_balance"] else 0
                trades = self._process_trades(data["trades"])
                today = datetime.now()
                last_day_of_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                days_left_in_month = max((last_day_of_month - today).days, 1)
                logger.info("Processing profit goal: ₹%.2f", target_profit)
                return self.suggest_profit_strategy(target_profit, 0.0, balance, trades, days_left_in_month)

            # Handle prediction accuracy query
            if any(term in user_query_lower for term in ["accuracy", "how accurate", "prediction accuracy"]):
                if not stock_name:
                    return "Please specify a stock (Reliance or Axis Bank) to check prediction accuracy."
                accuracy = self.get_prediction_accuracy(stock_name, specific_date)
                if accuracy["mean_absolute_error"] is None:
                    return f"No accuracy data available for {stock_name}{'' if not specific_date else ' on ' + specific_date}. Try another stock or date."
                response = (
                    f"Prediction accuracy for {stock_name}{'' if not specific_date else ' on ' + specific_date}: "
                    f"Mean Absolute Error: ₹{accuracy['mean_absolute_error']:.2f}, "
                    f"{accuracy['accuracy_percentage']}% of predictions were within 5% of actual prices."
                )
                response += "\nWant a chart of predictions vs. actual prices or trade details?"
                logger.info("Formatted prediction accuracy response")
                return response

            # Handle most profitable day/stock query
            if "most profitable day" in user_query_lower or "best day" in user_query_lower:
                trades = self._process_trades(data["trades"])
                return self.format_trade_response(trades, "most_profitable_day", specific_date, stock_name, days_back)
            if "most profitable stock" in user_query_lower or "best stock" in user_query_lower:
                trades = self._process_trades(data["trades"])
                return self.format_trade_response(trades, "most_profitable_stock", specific_date, stock_name, days_back)

            # Handle chart request
            if "chart" in user_query_lower or "graph" in user_query_lower:
                trades = self._process_trades(data["trades"])
                return self.generate_profit_chart(trades, days_back, stock_name)

            # Handle prediction query
            if any(term in user_query_lower for term in ["prediction", "forecast", "close price"]):
                latest_date = self.get_latest_prediction_date()
                if not latest_date:
                    logger.info("No predictions available")
                    return "No predictions available. Try updating the pipeline or specify a stock and date."
                
                formatted_latest_date = datetime.strptime(latest_date, "%Y-%m-%d").strftime("%B %d, %Y")
                
                if not stock_name:
                    logger.info("No stock specified for prediction")
                    return (
                        f"Predictions are available up to {formatted_latest_date}. "
                        f"Please specify a stock (Reliance or Axis Bank) or a date. Want to see trades instead?"
                    )

                if specific_date:
                    for pred in data["daily_predictions"]:
                        id, target_date, db_stock_name, predicted_price, actual_price, confidence_score = pred
                        if target_date == specific_date and stock_name.lower() in db_stock_name.lower():
                            formatted_date = datetime.strptime(specific_date, "%Y-%m-%d").strftime("%B %d, %Y")
                            actual_price_str = f"Actual price: ₹{actual_price:.2f}. " if actual_price is not None else ""
                            logger.info("Prediction found for %s on %s", stock_name, specific_date)
                            return (
                                f"For {stock_name} on {formatted_date}, I predict a price of around ₹{predicted_price:.2f}, "
                                f"with a confidence of {confidence_score*100:.1f}%. {actual_price_str}"
                                f"Want to see trades or a chart for that day?"
                            )
                    logger.info("No prediction for %s on %s", stock_name, specific_date)
                    return (
                        f"No prediction for {stock_name} on {datetime.strptime(specific_date, '%Y-%m-%d').strftime('%B %d, %Y')}. "
                        f"Latest prediction is for {formatted_latest_date}. Want predictions for that date or a chart?"
                    )

                # Fallback to latest prediction
                for pred in data["daily_predictions"]:
                    id, target_date, db_stock_name, predicted_price, actual_price, confidence_score = pred
                    if target_date == latest_date and stock_name.lower() in db_stock_name.lower():
                        formatted_date = datetime.strptime(latest_date, "%Y-%m-%d").strftime("%B %d, %Y")
                        actual_price_str = f"Actual price: ₹{actual_price:.2f}. " if actual_price is not None else ""
                        logger.info("Showing latest prediction for %s on %s", stock_name, latest_date)
                        return (
                            f"Latest prediction for {stock_name} on {formatted_date}: "
                            f"Predicted price ₹{predicted_price:.2f}, with {confidence_score*100:.1f}% confidence. "
                            f"{actual_price_str}Want to see trades or a chart for that day?"
                        )
                logger.info("No predictions for %s", stock_name)
                return f"No predictions for {stock_name}. Latest data is for {formatted_latest_date}. Try another stock or date."

            # Handle trade, profit, and loss queries
            trades = self._process_trades(data["trades"])
            trades.sort(key=lambda x: x["entry_time"])

            if days_match and any(term in user_query_lower for term in ["trade", "profit", "loss"]):
                num_days = int(days_match.group(1))
                last_trading_days = self.get_last_trading_days(num_days)
                if not last_trading_days:
                    logger.info("No trading days found for last %s days", num_days)
                    return f"No trading days found for the last {num_days} days. Want to check recent trades or predictions?"
                filtered_trades = [
                    t for t in trades
                    if t["entry_date"] in last_trading_days and (not stock_name or stock_name.lower() in t["stock"].lower())
                ]
                if "profit" in user_query_lower:
                    logger.info("Returning profitable trades for last %s days", num_days)
                    return self.format_trade_response(filtered_trades, "profit", specific_date, stock_name, num_days)
                if "loss" in user_query_lower:
                    logger.info("Returning losing trades for last %s days", num_days)
                    return self.format_trade_response(filtered_trades, "loss", specific_date, stock_name, num_days)
                logger.info("Returning trades for last %s days", num_days)
                return self.format_trade_response(filtered_trades, "last_x_days", specific_date, stock_name, num_days)

            if "last trade" in user_query_lower or "last one" in user_query_lower:
                logger.info("Returning last trade")
                return self.format_trade_response([trades[-1]] if trades else [], "last_one", specific_date, stock_name)
            if "date trade" in user_query_lower:
                logger.info("Returning trade dates")
                return self.format_trade_response(trades, "trade_dates", specific_date, stock_name)
            if "how many" in user_query_lower and stock_name:
                logger.info("Returning trade count")
                return self.format_trade_response(trades, "trade_count", specific_date, stock_name)
            if "intraday" in user_query_lower:
                intraday_trades = [t for t in trades if t["trade_type"].lower() == "intraday"]
                logger.info("Returning intraday trades")
                return self.format_trade_response(intraday_trades, "intraday", specific_date, stock_name)
            if "profit" in user_query_lower:
                logger.info("Returning profitable trades")
                return self.format_trade_response(trades, "profit", specific_date, stock_name)
            if "loss" in user_query_lower:
                logger.info("Returning losing trades")
                return self.format_trade_response(trades, "loss", specific_date, stock_name)

            logger.info("Returning general trade response")
            return self.format_trade_response(trades, "general", specific_date, stock_name)

        except Exception as e:
            logger.error("Query processing failed: %s", e)
            return f"Sorry, something went wrong: {str(e)}. Try asking about trades, predictions, or your balance."

    def run(self):
        """Run the pipeline in an interactive loop for testing."""
        try:
            logger.info("Starting interactive RAGPipeline")
            self.update_pipeline()
            print("Welcome to the Trading Chatbot! Type 'exit' to quit.")
            while True:
                user_query = input("Your query: ")
                if user_query.lower() == "exit":
                    logger.info("Exiting interactive mode")
                    break
                response = self.query(user_query)
                print(response)
        except Exception as e:
            logger.error("Interactive pipeline failed: %s", e)
            raise


if __name__ == "__main__":
    try:
        config = load_config()
        pipeline = RAGPipeline(config)
        pipeline.run()
    except Exception as e:
        logger.critical("Pipeline startup failed: %s", e)
        raise