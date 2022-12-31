import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf

# function to download and preprocess real-time data for Microsoft stock
def get_data():
  # use Yahoo Finance API to download real-time data for Microsoft stock
  msft = yf.Ticker("MSFT")
  df = msft.history(period="1d", interval="1m")

  # clean and format data
  df = df.dropna()
  df = df[['Close']]
  df = df.reset_index(drop=True)

  # normalize data
  df['Close'] = df['Close'] / df['Close'].iloc[0]

  return df

# function to split data into training and testing sets
def split_data(df, train_frac=0.8):
  # split data into training and testing sets
  train_size = int(len(df) * train_frac)
  train_data = df.iloc[:train_size]
  test_data = df.iloc[train_size:]

  # convert data to numpy arrays
  X_train = np.array(train_data['Close'])
  X_test = np.array(test_data['Close'])

  return (X_train, X_test)

# function to create and train RNN
def train_rnn(X_train, X_test):
  # create RNN model
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(units=50, input_shape=(1, 1)))
  model.add(tf.keras.layers.Dense(units=1))
  model.compile(loss='mean_squared_error', optimizer='adam')

  # reshape data for use with RNN
  X_train = X_train.reshape((X_train.shape[0], 1, 1))
  X_test = X_test.reshape((X_test.shape[0], 1, 1))

  # check shapes of input data
  print(f"X_train shape: {X_train.shape}")
  print(f"X_test shape: {X_test.shape}")

  # train RNN
  model.fit(X_train, X_train, epochs=100, batch_size=1, verbose=2)

  return model

# function to evaluate RNN
def evaluate_rnn(model, X_test):
  # make predictions with RNN
  predictions = model.predict(X_test)

  # calculate mean squared error
  mse = np.mean((predictions - X_test)**2)

  return mse

# function to make predictions with RNN
def predict_with_rnn(model, X_test):
  # make predictions with RNN
  predictions = model.predict(X_test)

  return predictions

def main():
  # download and preprocess data
  df = get_data()

  # split data into training and testing sets
  X_train, X_test = split_data(df)

  # create and train RNN
  model = train_rnn(X_train, X_test)

  # evaluate RNN
  mse = evaluate_rnn(model, X_test)
  print(f"Mean Squared Error: {mse}")

  # make predictions with RNN
  predictions = predict_with_rnn(model, X_test)
  print(predictions)

# call main function
if __name__ == "__main__":
  main()
