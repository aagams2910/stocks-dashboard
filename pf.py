import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate sector allocation
def get_sector_allocation(tickers):
    sector_allocation = {}
    for ticker in tickers:
        stock = yf.Ticker(f"{ticker}.NS")
        sector = stock.info.get("sector", "Unknown")
        sector_allocation[sector] = sector_allocation.get(sector, 0) + 1
    return sector_allocation

# Function to calculate correlation matrix
def get_correlation_matrix(tickers):
    data = yf.download([f"{ticker}.NS" for ticker in tickers], period="1y")["Close"]
    correlation_matrix = data.corr()
    return correlation_matrix

# Function to calculate moving averages
def calculate_moving_averages(ticker, window_short=50, window_long=200):
    data = yf.Ticker(f"{ticker}.NS").history(period="1y")
    data["MA_Short"] = data["Close"].rolling(window=window_short).mean()
    data["MA_Long"] = data["Close"].rolling(window=window_long).mean()
    return data

# Initialize portfolio in session state if it doesn't exist
if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = {"Ticker": [], "Shares": [], "Purchase Price": []}

if "last_refresh_time" not in st.session_state:
    st.session_state["last_refresh_time"] = 0

# Predefined list of NSE stock tickers
# Updated list of verified NSE tickers
nse_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
               "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "LT.NS", 
               "ITC.NS", "ASIANPAINT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS"]
# Function to check if ticker is valid
@st.cache_data
def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Check multiple validity indicators
        if not stock.info or 'regularMarketPrice' not in stock.info:
            return False
        # Verify historical data availability
        hist = stock.history(period="1d")
        return not hist.empty
    except Exception as e:
        st.warning(f"Validation failed for {ticker}: {str(e)}")
        return False
# Function to add stock to portfolio
def add_stock_to_portfolio(ticker, shares, purchase_price=None):
    if ticker in st.session_state["portfolio"]["Ticker"]:
        index = st.session_state["portfolio"]["Ticker"].index(ticker)
        st.session_state["portfolio"]["Shares"][index] += shares
    else:
        st.session_state["portfolio"]["Ticker"].append(ticker)
        st.session_state["portfolio"]["Shares"].append(shares)
        if purchase_price is None:
            purchase_price = yf.Ticker(f"{ticker}.NS").history(period="1d")["Close"].iloc[-1]
        st.session_state["portfolio"]["Purchase Price"].append(purchase_price)

# Function to refresh stock data
def refresh_stock_data():
    st.session_state["last_refresh_time"] = 0
    st.rerun()

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(df):
    df["Current Price"] = [yf.Ticker(f"{ticker}.NS").history(period="1d")["Close"].iloc[-1] for ticker in df["Ticker"]]
    df["Market Value"] = df["Shares"] * df["Current Price"]
    df["Total Investment"] = df["Shares"] * df["Purchase Price"]
    df["Total Profit/Loss"] = df["Market Value"] - df["Total Investment"]
    df["Return (%)"] = ((df["Total Profit/Loss"] / df["Total Investment"]) * 100).round(2)
    
    # Add Profit/Loss Label
    df["Status"] = df["Total Profit/Loss"].apply(lambda x: "Profit" if x > 0 else "Loss")
    
    return df


# Function to predict stock price using linear regression
def predict_stock_price(ticker):
    data = yf.Ticker(f"{ticker}.NS").history(period="1y")
    if data.empty:
        return None
    data["Date"] = data.index
    data["Days"] = (data["Date"] - data["Date"].min()).dt.days
    X = data[["Days"]]
    y = data["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_days = data["Days"].max() + 30  # Predict 30 days into the future
    future_price = model.predict([[future_days]])
    return future_price[0]

def get_nse_top_gainers():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-market-indices'
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com/", headers=headers, timeout=10)
        
        # Updated working API endpoint for gainers
        response = session.get(
            "https://www.nseindia.com/api/live-analysis-variations?index=gainers",
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Process valid response
        df = pd.DataFrame(data['data'])[['symbol', 'lastPrice', 'pChange']]
        df.columns = ['Symbol', 'Last Price (₹)', '% Change']
        return df.head(10).reset_index(drop=True)

    except Exception as e:
        st.error(f"Error fetching NSE top gainers: {str(e)}")
        return pd.DataFrame(columns=['Symbol', 'Last Price (₹)', '% Change'])

def get_nse_top_losers():
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-market-indices'
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com/", headers=headers, timeout=10)
        
        # Updated working API endpoint for losers
        response = session.get(
            "https://www.nseindia.com/api/live-analysis-variations?index=losers",
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Process valid response
        df = pd.DataFrame(data['data'])[['symbol', 'lastPrice', 'pChange']]
        df.columns = ['Symbol', 'Last Price (₹)', '% Change']
        return df.head(10).reset_index(drop=True)

    except Exception as e:
        st.error(f"Error fetching NSE top losers: {str(e)}")
        return pd.DataFrame(columns=['Symbol', 'Last Price (₹)', '% Change'])    

# App layout
st.title("Stocks Investing Dashboard")
st.sidebar.title("Portfolio Management")
menu = st.sidebar.radio("Choose an option:", ["Add Stock", "View Portfolio", "Analyze Portfolio", "AI Insights", "Market Overview", "News", "Help"])
# Refresh Button
if st.sidebar.button("Refresh Data"):
    refresh_stock_data()

# Reset Portfolio
if st.sidebar.button("Reset Portfolio"):
    st.session_state["portfolio"] = {"Ticker": [], "Shares": [], "Purchase Price": []}
    st.success("Portfolio reset successfully!")

# Save and Load Portfolio
if menu == "View Portfolio":
    st.subheader("Save/Load Portfolio")
    if st.button("Save Portfolio"):
        with open("portfolio.json", "w") as f:
            json.dump(st.session_state["portfolio"], f)
        st.success("Portfolio saved successfully!")

    if st.button("Load Portfolio"):
        try:
            with open("portfolio.json", "r") as f:
                st.session_state["portfolio"] = json.load(f)
            st.success("Portfolio loaded successfully!")
        except FileNotFoundError:
            st.warning("No saved portfolio found!")

# Add Stock Page
if menu == "Add Stock":
    st.subheader("Add Stock to Portfolio")

    ticker = st.text_input("Enter Stock Ticker (e.g., TCS):").upper()
    shares = st.number_input("Enter Number of Shares:", min_value=1, value=1, step=1)
    avg_purchase_price = st.number_input("Enter Average Purchase Price (₹):", min_value=0.0, value=0.0, step=0.01)

    if st.button("Add to Portfolio"):
        if ticker and is_valid_ticker(ticker):
            if avg_purchase_price == 0.0:  # If the user doesn't enter a price, fetch the latest price
                avg_purchase_price = yf.Ticker(f"{ticker}.NS").history(period="1d")["Close"].iloc[-1]
            
            add_stock_to_portfolio(ticker, shares, avg_purchase_price)
            st.success(f"Added {shares} shares of {ticker} at ₹{avg_purchase_price:.2f} to your portfolio.")
        else:
            st.warning("Invalid ticker! Please enter a valid stock ticker.")

if st.button("Add Random NSE Stocks"):
    random_stocks = random.sample(nse_tickers, 5)
    for stock in random_stocks:
        try:
            # Fetch data and check if available
            stock_data = yf.Ticker(f"{stock}.NS").history(period="1d")
            if stock_data.empty:
                st.warning(f"No data available for {stock}. Skipping.")
                continue
            shares = random.randint(1, 50)
            avg_purchase_price = stock_data["Close"].iloc[-1]
            add_stock_to_portfolio(stock, shares, avg_purchase_price)
        except Exception as e:
            st.warning(f"Error adding {stock}: {str(e)}. Skipping.")
            continue
    st.success("Random NSE stocks added to your portfolio!")
    st.write("### Current Portfolio")
    st.table(pd.DataFrame(st.session_state["portfolio"]))

#View portfolio
if menu == "View Portfolio":
    st.subheader("Current Portfolio with Market Value and Profit/Loss")

    if st.session_state["portfolio"]["Ticker"]:
        df = pd.DataFrame(st.session_state["portfolio"])
        df = calculate_portfolio_metrics(df)

        # Sort by profit percentage in ascending order
        df = df.sort_values(by="Return (%)", ascending=True)

        # Display the table without index
        st.table(df.reset_index(drop=True))

        total_investment = df["Total Investment"].sum()
        total_profit_loss = df["Total Profit/Loss"].sum()
        overall_return = (total_profit_loss / total_investment) * 100

        st.write(f"### Total Investment: ₹{total_investment:,.2f}")
        st.write(f"### Total Profit/Loss: ₹{total_profit_loss:,.2f}")
        st.write(f"### Overall Return: {overall_return:.2f}%")

        # Portfolio Visualization
        st.subheader("Portfolio Distribution")
        fig = px.pie(df, names='Ticker', values='Market Value', title="Portfolio Distribution")
        st.plotly_chart(fig)

        # Historical Performance Chart
        st.subheader("Historical Portfolio Performance")
        portfolio_value_over_time = pd.DataFrame()
        for ticker in df["Ticker"]:
            stock_data = yf.Ticker(f"{ticker}.NS").history(period="1y")
            portfolio_value_over_time[ticker] = stock_data["Close"] * df[df["Ticker"] == ticker]["Shares"].values[0]
        portfolio_value_over_time["Total"] = portfolio_value_over_time.sum(axis=1)
        fig = px.line(portfolio_value_over_time, y="Total", title="Portfolio Value Over Time")
        st.plotly_chart(fig)

        # Sector Allocation
        st.subheader("Sector Allocation")
        sector_allocation = get_sector_allocation(df["Ticker"])
        fig = px.pie(names=list(sector_allocation.keys()), values=list(sector_allocation.values()), title="Sector Allocation")
        st.plotly_chart(fig)

        # Risk Analysis (Volatility)
        st.subheader("Risk Analysis (Volatility)")
        volatility_data = {}
        for ticker in df["Ticker"]:
            stock_data = yf.Ticker(f"{ticker}.NS").history(period="1y")
            volatility_data[ticker] = stock_data["Close"].std()
        fig = px.bar(x=list(volatility_data.keys()), y=list(volatility_data.values()), labels={"x": "Ticker", "y": "Volatility"}, title="Stock Volatility")
        st.plotly_chart(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        correlation_matrix = get_correlation_matrix(df["Ticker"])
        fig = px.imshow(correlation_matrix, text_auto=True, title="Correlation Between Stocks")
        st.plotly_chart(fig)

        # Moving Averages for Individual Stocks
        st.subheader("Moving Averages for Individual Stocks")
        selected_ticker = st.selectbox("Select a stock to view moving averages:", df["Ticker"])
        moving_averages_data = calculate_moving_averages(selected_ticker)
        fig = px.line(moving_averages_data, y=["Close", "MA_Short", "MA_Long"], title=f"{selected_ticker} Moving Averages")
        st.plotly_chart(fig)

        # Stock Removal
        st.subheader("Remove Stock from Portfolio")
        ticker_to_remove = st.selectbox("Select a stock to remove:", df["Ticker"])
        if st.button("Remove Stock"):
            index = df["Ticker"].tolist().index(ticker_to_remove)
            for key in st.session_state["portfolio"]:
                st.session_state["portfolio"][key].pop(index)
            st.success(f"Removed {ticker_to_remove} from your portfolio.")
            st.rerun()


    else:
        st.warning("Your portfolio is empty! Add some stocks first.")

# Analyze Portfolio Page
elif menu == "Analyze Portfolio":
    st.subheader("Candlestick Chart for Individual Stock")

    if st.session_state["portfolio"]["Ticker"]:
        selected_ticker = st.selectbox("Select a stock to analyze:", st.session_state["portfolio"]["Ticker"])
        timeframe_options = {
            "1 Day": "1d",
            "1 Week": "7d",
            "1 Month": "1mo",
            "1 Year": "1y",
            "5 Years": "5y",
            "Max": "max"
        }
        selected_timeframe = st.selectbox("Select Timeframe:", list(timeframe_options.keys()))

        if selected_ticker and selected_timeframe:
            ticker_data = yf.Ticker(f"{selected_ticker}.NS").history(period=timeframe_options[selected_timeframe])

            if not ticker_data.empty:
                # Remove rows where any of the candlestick data is missing (NaN values)
                ticker_data = ticker_data.dropna(subset=['Open', 'High', 'Low', 'Close'])

                fig = go.Figure(data=[go.Candlestick(
                    x=ticker_data.index,
                    open=ticker_data['Open'],
                    high=ticker_data['High'],
                    low=ticker_data['Low'],
                    close=ticker_data['Close']
                )])
                fig.update_layout(
                    title=f"{selected_ticker} Candlestick Chart ({selected_timeframe})", 
                    xaxis_title="Date", 
                    yaxis_title="Price (₹)"
                )
                st.plotly_chart(fig)
            else:
                st.warning("No data available for the selected timeframe.")
    else:
        st.warning("Your portfolio is empty! Add some stocks first.")
elif menu == "Market Overview":
    st.subheader("Top Gainers & Losers - NSE")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔼 Top 10 Gainers")
        gainers = get_nse_top_gainers()
        if not gainers.empty:
            st.dataframe(gainers.style.format({
                'Last Price (₹)': '{:.2f}',
                '% Change': '{:.2f}%'
            }), height=400)
        else:
            st.warning("Failed to fetch NSE gainers data")

    with col2:
        st.markdown("### 🔽 Top 10 Losers")
        losers = get_nse_top_losers()
        if not losers.empty:
            st.dataframe(losers.style.format({
                'Last Price (₹)': '{:.2f}',
                '% Change': '{:.2f}%'
            }), height=400)
        else:
            st.warning("Failed to fetch NSE losers data")

    st.markdown("---")
    st.write("Note: BSE data is currently not available due to API limitations")# AI Insights Page
elif menu == "AI Insights":
    st.subheader("AI-Powered Stock Insights")

    if st.session_state["portfolio"]["Ticker"]:
        selected_ticker = st.selectbox("Select a stock for AI insights:", st.session_state["portfolio"]["Ticker"])
        
        if st.button("Predict Future Price"):
            # Fetch historical data
            data = yf.Ticker(f"{selected_ticker}.NS").history(period="1y")
            
            if not data.empty:
                # Prepare data for linear regression
                data["Date"] = data.index
                data["Days"] = (data["Date"] - data["Date"].min()).dt.days
                X = data[["Days"]]
                y = data["Close"]
                
                # Train the model
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict future price (30 days into the future)
                future_days = data["Days"].max() + 30
                future_price = model.predict([[future_days]])
                
                # Generate predictions for the entire range (historical + future)
                all_days = np.arange(data["Days"].min(), future_days + 1).reshape(-1, 1)
                all_predictions = model.predict(all_days)
                
                # Create a DataFrame for plotting
                plot_data = pd.DataFrame({
                    "Date": pd.date_range(start=data["Date"].min(), periods=len(all_days)),
                    "Predicted Price": all_predictions
                })
                
                # Plot historical and predicted prices
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data["Date"],
                    y=data["Close"],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Predicted data
                fig.add_trace(go.Scatter(
                    x=plot_data["Date"],
                    y=plot_data["Predicted Price"],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='red', dash='dash')
                ))
                
                # Highlight the future prediction point
                fig.add_trace(go.Scatter(
                    x=[plot_data["Date"].iloc[-1]],
                    y=[future_price[0]],
                    mode='markers',
                    name='Future Prediction',
                    marker=dict(color='green', size=10)
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_ticker} Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    legend_title="Legend",
                    showlegend=True
                )
                
                # Display the plot
                st.plotly_chart(fig)
                
                # Display the predicted price
                st.success(f"Predicted price for {selected_ticker} in 30 days: ₹{future_price[0]:.2f}")
            else:
                st.warning("Unable to predict price for the selected stock.")
    else:
        st.warning("Your portfolio is empty! Add some stocks first.")
# News Page

# Help Page
elif menu == "Help":
    st.subheader("How to Use This App")
    st.write("""
    1. **Add Stock**: Enter a stock ticker and the number of shares to add it to your portfolio.
    2. **View Portfolio**: See your current portfolio, including market value and profit/loss.
    3. **Analyze Portfolio**: Analyze individual stock performance using candlestick charts.
    4. **AI Insights**: Get AI-powered predictions and insights for your stocks.
    5. **News**: Stay updated with the latest financial news.
    """)
