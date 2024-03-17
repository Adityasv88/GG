import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import statsmodels.api as sm

# Suppressing warnings
warnings.filterwarnings('ignore')

# Function to check stationarity
def test_stationarity(daf, var):
    daf['rollMean'] = daf[var].rolling(window=12).mean()
    daf['rollStd'] = daf[var].rolling(window=12).std()
    adfTest = adfuller(daf[var], autolag='AIC')
    stats = pd.Series(adfTest[0:4], index=['Test Statistics', 'p-value', 'Lags', 'No. of Observations'])
    st.write("### Rolling Statistics Test")
    st.write(stats)
    for key, values in adfTest[4].items():
        st.write('Critical Values', key, ":", values)
    st.write("#### Plot of", var)
    fig, ax = plt.subplots()
    ax.plot(daf.index, daf[var], label=var)
    ax.plot(daf.index, daf['rollMean'], label='Rolling Mean')
    ax.plot(daf.index, daf['rollStd'], label='Rolling Std')
    ax.legend()
    st.pyplot(fig)

# Fit ARIMA model
def fit_arima(data):
    arima_model = ARIMA(data, order=(1, 1, 1))
    arima_result = arima_model.fit()
    prediction = arima_result.predict(start=data.index[0], end=data.index[-1])
    st.subheader("ARIMA Model")
    st.write("#### Actual vs. Predicted (ARIMA)")
    fig, ax = plt.subplots()
    ax.plot(data.index, data, label='Actual')
    ax.plot(data.index, prediction, label='Predicted')
    ax.legend()
    st.pyplot(fig)

# Fit SARIMA model
def fit_sarima(data):
    sarima_model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_result = sarima_model.fit()
    prediction = sarima_result.predict(start=data.index[0], end=data.index[-1])
    st.subheader("SARIMA Model")
    st.write("#### Actual vs. Predicted (SARIMA)")
    fig, ax = plt.subplots()
    ax.plot(data.index, data, label='Actual')
    ax.plot(data.index, prediction, label='Predicted')
    ax.legend()
    st.pyplot(fig)

# Fit ARIMA and SARIMA models
def fit_models(data):
    fit_arima(data)
    fit_sarima(data)

# Main function
def main():
    st.title("Stock Analysis")
    st.write("""
    Welcome to the Stock Analysis App. This app analyzes historical stock data using various time series analysis techniques.
    """)

    # Read data
    df = pd.read_csv('IBN.csv', parse_dates=['Date'], index_col='Date')

    # Display original data
    st.subheader("Original Data")
    st.write(df)

    # Plot original data
    st.subheader("Original Data Plot")
    st.write("""
    The original data plot shows the closing prices of the stock over time.
    """)
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Adj Close'])
    st.pyplot(fig)

    # Decompose time series
    df1 = sm.tsa.seasonal_decompose(df['Adj Close'], model='additive', period=12)

    # Display decomposed data
    st.subheader("Decomposed Data")
    st.write("""
    The decomposed data consists of trend, seasonal, and residual components obtained using the seasonal decomposition of time series (STL) method.
    """)
    st.write("#### Trend Component")
    st.write("The trend component represents the underlying trend in the data.")
    fig, ax = plt.subplots()
    ax.plot(df1.trend)
    st.pyplot(fig)
    st.write("#### Seasonal Component")
    st.write("The seasonal component captures the periodic patterns in the data.")
    fig, ax = plt.subplots()
    ax.plot(df1.seasonal)
    st.pyplot(fig)
    st.write("#### Residual Component")
    st.write("The residual component contains the irregular fluctuations not explained by the trend and seasonal components.")
    fig, ax = plt.subplots()
    ax.plot(df1.resid)
    st.pyplot(fig)

    # Perform rolling statistics test
    st.subheader("Rolling Statistics Test")
    st.write("""
    The rolling statistics test is performed to check the stationarity of the time series data. It involves calculating the rolling mean and rolling standard deviation and comparing them with the critical values.
    """)
    test_stationarity(df, 'Adj Close')

    # Fit ARIMA and SARIMA models
    st.subheader("Model Fitting and Predictions")
    st.write("""
    ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal AutoRegressive Integrated Moving Average) models are fitted to the data to make predictions. The actual vs. predicted values are visualized to assess the model performance.
    """)
    fit_models(df['Adj Close'])

    # Autocorrelation graph
    st.subheader("Autocorrelation Graph")
    st.write("""
    The autocorrelation graph shows the correlation between a variable and its lagged values. It helps in identifying the presence of autocorrelation in the time series data.
    """)
    fig, ax = plt.subplots()
    plot_acf(df['Adj Close'], lags=40, ax=ax)
    st.pyplot(fig)

    # Partial autocorrelation plot
    st.subheader("Partial Autocorrelation Plot")
    st.write("""
    The partial autocorrelation plot displays the correlation between a variable and its lagged values after removing the effects of intermediate lagged variables. It is useful in determining the order of the ARIMA model.
    """)
    fig, ax = plt.subplots()
    plot_pacf(df['Adj Close'], lags=40, ax=ax)
    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
