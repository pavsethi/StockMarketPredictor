import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

with open('data/ticker_symbols.json', 'r') as file:
    ticker_data = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

# API route to filter ticker symbols based on user input
@app.route('/search-tickers', methods=['GET'])
def search_tickers():
    query = request.args.get('query', '').upper()
    
    # Filter the ticker data based on the user's input
    # Iterate over the values of the JSON (ignoring the "0", "1" keys)
    filtered_symbols = [
        item['ticker'] for item in ticker_data.values()
        if query in item.get('ticker', '').upper() or query in item.get('title', '').upper()
    ]
    
    return jsonify(filtered_symbols)

# Function to fetch the current stock price
def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='1d')['Close'].iloc[-1]

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['symbol']

    # Fetch current price
    try:
        current_price = get_current_price(stock_symbol)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
    # Fetch historical stock data
    data = yf.download(stock_symbol, start="2020-01-01")
    
    # Prepare data for training the model
    data['Date'] = pd.to_datetime(data.index)
    data['Date'] = data['Date'].map(pd.Timestamp.timestamp)
    X = np.array(data['Date']).reshape(-1, 1)
    y = np.array(data['Close']).reshape(-1, 1)

    # Train the model using linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions for the next 30 days
    last_date = data.index[-1]
    future_dates = pd.date_range(last_date, periods=30).map(pd.Timestamp.timestamp).values.reshape(-1, 1)
    predicted_prices = model.predict(future_dates)

    # Plot the predictions
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label="Historical Prices")
    future_dates = pd.date_range(last_date, periods=30)
    plt.plot(future_dates, predicted_prices, label="Predicted Prices", linestyle="--")
    plt.title(f"Stock Price Prediction for {stock_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # Save the plot as a PNG file
    plt.savefig('static/prediction_plot.png')
    plt.close()

    # Convert prediction to a readable format
    predictions = predicted_prices.flatten().tolist()
    future_dates = future_dates.strftime('%Y-%m-%d').tolist()

    return jsonify({"dates": future_dates, "prices": predictions, "current_price": current_price})

if __name__ == '__main__':
    app.run(debug=True)
