# XAI Trader

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## Abstract

**XAI Trader** is a personal project designed to provide data-driven stock market insights. It predicts next-day closing prices for **Axis Bank** and **Reliance Industries** using a GRU-based deep learning model. The system offers a user-friendly interface to simulate trades, track performance, and query trading activity via natural language.

## Overview

Stock price prediction is complex due to market volatility. XAI Trader delivers accurate next-day closing price forecasts, a conversational chatbot, and interactive visualizations. Built with **FastAPI** for the backend and **Streamlit** for the frontend, it’s ideal for exploring stock predictions and trade simulations.

### Problem Statement & Goal

The goal is to provide reliable stock price predictions with transparency and interactivity. XAI Trader combines GRU-based forecasts with a natural language interface and visualizations to help users understand model performance and simulate trades.

### Target Audience

- Individual traders exploring predictions in a risk-free environment.
- Finance learners studying market trends and ML applications.
- ML enthusiasts interested in time series forecasting and RAG-based systems.

## Key Features

- **Accurate Predictions**: GRU model forecasts next-day closing prices for Axis Bank and Reliance Industries, stored in SQLite.
- **Natural Language Queries**: Chatbot powered by Gemini 1.5 Flash and RAG answers questions like “What’s tomorrow’s prediction?” or “What’s my balance?”
- **Trade Simulation**: Simulate buy/sell trades and track profits/losses.
- **Performance Visualization**: Plotly charts show predicted vs. actual prices over 30 days, with 7-day and 14-day mean error trends.
- **User-Friendly UI**: Streamlit frontend for an intuitive experience.
- **Automated Scheduler**: Updates predictions and data daily at 8 PM IST.

## Technologies & Methodologies

- **Models & Embeddings**:
  - **GRU**: For time series forecasting.
  - **Sentence Transformers (all-MiniLM-L6-v2)**: For RAG embeddings.
- **Databases**:
  - **SQLite**: Stores stock data, trades, and balances.
  - **ChromaDB**: Vector storage for RAG queries.
- **LLM & RAG**: Gemini 1.5 Flash API with Retrieval-Augmented Generation.
- **Visualization**: Plotly for dynamic charts.
- **Frameworks**:
  - **FastAPI**: Backend API for data processing.
  - **Streamlit**: Frontend for user interaction.
- **Tools**: yfinance for stock data, SHAP for model interpretability, Docker for containerization.

## Project Workflow

1. **Data Collection**: Historical stock data (2020–2025) for Axis Bank and Reliance Industries via yfinance.
2. **Feature Engineering**: Price-based features (Open, High, Low, Close, Volume) and technical indicators (e.g., moving averages).
3. **Model Training**: GRU model predicts next-day closing prices.
4. **Data Storage**: Predictions and trades in SQLite; embeddings in ChromaDB.
5. **Backend**: FastAPI serves API endpoints for predictions and queries.
6. **Frontend**: Streamlit displays charts, trade simulations, and chatbot.
7. **Scheduler**: Daily updates at 8 PM IST using `supervisord`.

## Data Understanding

### Data Sources
- **Source**: yfinance library for 5 years of stock data (2020–2025) in CSV format.
- **Attributes**: Date, Open, High, Low, Close, Volume.

### Feature Analysis
- **Features Used**: Price-based (High, Low, Close) and select indicators (e.g., moving averages).
- **Correlation Analysis**: High, Low, and Close strongly correlated with today’s and next day’s prices; complex indicators (e.g., RSI, MACD) excluded due to low predictive power.
- **SHAP Analysis**: Validated feature importance for model transparency.

## Getting Started

### Prerequisites
- Python 3.9+
- Docker
- Git

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shahabzack/XAI_TraderX.git
   cd XAI_TraderX
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your values (e.g., API keys, database URL). Example:
     ```
     STOCK_API_KEY=your_stock_api_key_here
     GEMINI_API_KEY=your_gemini_api_key_here
     DB_URL=sqlite:///database/traderx.db
     ```
4. **Run Locally**:
   - **Option 1: Without Docker**:
     - Start the backend (FastAPI):
       ```bash
       uvicorn app.main:app --reload
       ```
     - Launch the frontend (Streamlit):
       ```bash
       streamlit run app/frontend.py
       ```
   - **Option 2: With Docker**:
     ```bash
     docker build -t xai-trader:latest .
     docker run -p 80:80 --env-file .env xai-trader:latest
     ```
5. **Access the App**:
   - Open `http://localhost:8501` for Streamlit (frontend).
   - Open `http://localhost:8000/docs` for FastAPI (API documentation).

## Usage
- **Chatbot**: Query predictions (e.g., “What’s the prediction for Reliance tomorrow?”), trades, or balances via Streamlit.
- **Trade Simulation**: Simulate buy/sell trades and track profits/losses.
- **Visualizations**: View 30-day predicted vs. actual prices and 7/14-day error trends.
- **Scheduler**: Predictions update daily at 8 PM IST.

## Scope & Limitations
### Included
- Next-day price predictions for Axis Bank and Reliance Industries.
- GRU model with SHAP analysis.
- SQLite and ChromaDB storage.
- FastAPI/Streamlit with RAG-based chatbot.
- Trade simulation and visualizations.

### Not Included
- Real-time or intraday predictions.
- Live trading platform integration.
- Multi-stock portfolio analysis.
- News/sentiment-based predictions.

## Future Improvements
- Support for additional stocks.
- Real-time data integration.
- Integration of reinforcement learning for adaptive trading strategies.
- Enhanced RAG with larger LLMs for improved chatbot responses.

## License
This project is licensed under the [MIT License](LICENSE).

## Portfolio & Contact
Explore more projects at [https://shahabzack.github.io/Ds_portfolio/](https://shahabzack.github.io/Ds_portfolio/).

Thank you for exploring **XAI Trader** — an intelligent stock trading assistant!