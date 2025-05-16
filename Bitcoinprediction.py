#!/usr/bin/env python3
"""Download Bitcoin prices and forecast using three models."""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA


def download_data(ticker="BTC-USD", period="5y"):
    """Download historical data from Yahoo Finance."""
    print(f"Downloading {ticker} data for {period}...")
    df = yf.download(ticker, period=period)
    print("Data downloaded:\n", df.head())
    return df


def prepare_random_forest_data(df):
    """Create features and next-day target for RandomForest."""
    print("Preparing data for RandomForest...")
    df = df.copy()
    df["target"] = df["Close"].shift(-1)
    df = df.dropna()
    features = ["Open", "High", "Low", "Close", "Volume"]
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    test_dates = y_test.index
    return X_train, X_test, y_train, y_test, test_dates


def run_random_forest(X_train, X_test, y_train, y_test):
    print("\n=== Training RandomForestRegressor ===")
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print(f"Random Forest RMSE: {rmse:.2f}")
    print("RandomForest training complete!")
    return model, preds, rmse


def prepare_lstm_data(df, look_back=3):
    """Prepare sequences for LSTM model."""
    print("Preparing data for LSTM...")
    values = df[["Close"]].values.astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(scaled) - look_back - 1):
        X.append(scaled[i:(i + look_back), 0])
        y.append(scaled[i + look_back, 0])
    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * 0.67)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # reshape input to be [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return X_train, X_test, y_train, y_test, scaler


def run_lstm(X_train, X_test, y_train, y_test, scaler):
    print("\n=== Training LSTM ===")
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(1, X_train.shape[2])))
    model.add(LSTM(256))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, y_train, epochs=50, batch_size=50, verbose=1, shuffle=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
    print(f"LSTM Train RMSE: {train_rmse:.4f}")
    print(f"LSTM Test RMSE: {test_rmse:.4f}")
    print("LSTM training complete!")
    return model, test_pred.flatten(), test_rmse


def prepare_arima_data(series, test_size=0.2):
    """Split series for ARIMA training and testing."""
    print("Preparing data for ARIMA...")
    split = int(len(series) * (1 - test_size))
    train, test = series[:split], series[split:]
    return train, test


def run_arima(train_series, test_series):
    print("\n=== Training ARIMA(5,1,0) ===")
    model = ARIMA(train_series, order=(5, 1, 0))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=len(test_series))
    rmse = mean_squared_error(test_series, preds) ** 0.5
    print(f"ARIMA RMSE: {rmse:.2f}")
    print("ARIMA training complete!")
    return model_fit, preds, rmse


def main():
    print("Starting Bitcoin prediction pipeline...")
    df = download_data()

    # Random Forest
    X_train, X_test, y_train, y_test, test_dates = prepare_random_forest_data(df)
    rf_model, rf_preds, rf_rmse = run_random_forest(X_train, X_test, y_train, y_test)

    # LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler = prepare_lstm_data(df)
    lstm_model, lstm_preds_scaled, lstm_rmse = run_lstm(
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler
    )
    lstm_preds_all = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()

    # Align LSTM output length with RandomForest test set
    align_len = min(len(test_dates), len(lstm_preds_all))
    lstm_preds = lstm_preds_all[-align_len:]
    test_dates = test_dates[-align_len:]
    y_test = y_test[-align_len:]

    # ARIMA
    train_series, test_series = prepare_arima_data(df["Close"])
    arima_model, arima_preds, arima_rmse = run_arima(train_series, test_series)

    # Assemble comparison table
    compare_df = pd.DataFrame({
        "Date": test_dates,
        "Actual": y_test.values,
        "RandomForest": rf_preds[-align_len:],
        "LSTM": lstm_preds,
        "ARIMA": arima_preds.values[-align_len:],
    })

    print("\n=== Prediction Comparison (last 5 points) ===")
    print(compare_df.tail(5).to_string(index=False))

    print("\n=== RMSE Report ===")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
    print(f"LSTM Test RMSE: {lstm_rmse:.2f}")
    print(f"ARIMA RMSE: {arima_rmse:.2f}")
    print("Pipeline complete!")


if __name__ == "__main__":
    main()
