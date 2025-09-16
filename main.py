# app.py

import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Streamlit page settings
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title(" Stock Price Predictor")
st.markdown("Enter a stock symbol (like **TCS.NS**, **RELIANCE.NS**, or **PRESTIGE.NS**) to get the next 3 days' predicted closing prices using Random Forest.")

# User input
symbol = st.text_input("Stock Symbol", "PRESTIGE.NS")

# Predict button
if st.button("Predict"):
    try:
        # Load last 60 days of data for better training
        end_date = datetime.today()
        start_date = end_date - timedelta(days=60)
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            st.warning("Could not fetch stock data. Please check the symbol.")
        else:
            data = data[['Close']]
            data['Prev_Close'] = data['Close'].shift(1)
            data.dropna(inplace=True)

            # Prepare data
            X = data[['Prev_Close']]
            y = data['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Regression Fit Graph
            st.subheader("Fit: Prev Close vs Actual Close")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.scatter(X_test, y_test, color='blue', label='Actual Price')
            ax1.scatter(X_test, model.predict(X_test), color='red', label='Predicted Price')
            ax1.set_xlabel("Previous Day's Close")
            ax1.set_ylabel("Today's Close")
            ax1.set_title("Random Forest Predictions on Test Data")
            ax1.grid(True)
            ax1.legend()
            st.pyplot(fig1)

            # Predict next 3 days
            last_close = data['Close'].iloc[-1]
            future_dates = []
            predictions = []
            for i in range(3):
                next_price = float(model.predict(np.array([last_close]).reshape(1, -1))[0])
                next_price = round(next_price, 2)
                last_close = next_price
                future_dates.append((end_date + timedelta(days=i + 1)).strftime('%Y-%m-%d'))
                predictions.append(next_price)

            # Show predictions
            st.success("Prediction Complete!")
            for d, p in zip(future_dates, predictions):
                st.write(f" **{d}**: ₹{p}")

            # Forecast plot
            st.subheader(f" 3-Day Forecast for {symbol}")
            fig2, ax2 = plt.subplots()
            ax2.plot(future_dates, predictions, marker='o', linestyle='--', color='green')
            ax2.set_title("Next 3-Day Price Forecast (Random Forest)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Predicted Close Price (INR)")
            ax2.grid(True)
            st.pyplot(fig2)

    except Exception as e:
        st.error(f" Error: {e}")
