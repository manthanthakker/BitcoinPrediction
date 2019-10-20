# BitcoinPrediction

### CryptoCurrency prediction using Deep Recurrent Neural Networks
This repository contains various Machine learning models used in industry to predict stock prices and cryptocurrency in finance industry. 

  - Fundamental analysis of the stock price using Yahoo Finance
  - Data Visualization using Seaborn
  - ARIMA model to capture the trends,seasonality, forecast the prices and use as a baseline
  - Simpler machine learning models (Random Forest, Regression etc)
  - Recurrent Neural Networks / Long Short Term Memory Networks

Each model is compared against each other to highlight pros and cons of each model. 

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [fastai]
- [pytorch]

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

### Code

The source code is divided into multiple sections following the machine learning design pattern of : Data Exploration, Training, Testing and Hyperparameter Optimization.
You can view the precompiled version of the notebook or you can rerun the entire notebook. The datasets are made available on public S3 Buckets. 
Running the notebook, will automatically download the datasets for you. 

### Run

In a terminal or command window, navigate to the top-level project directory `boston_housing/` (that contains this README) and run one of the following commands:

```bash
ipython notebook BitcoinPredictionRNN.ipynb
```  
or
```bash
jupyter notebook BitcoinPredictionRNN.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Data
BitCoin Price Data from Jan 2015- August 2018. The prices are as per coinbase cryptoexchange. There were many missing values and forward strategy was used to fill these missing values. 

**Features**
BitCoin Price Data from Jan 2015- August 2018 

**Target Variable**
 `Close Price`: Close price of Bitcoin for each day
