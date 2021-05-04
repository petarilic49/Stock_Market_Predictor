#PURPOSE: to investigate the different machine learning regression algorithms and see which is best for predicting stock market prices
import numpy as  np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#We will use AMD stock prices for example
#Load the stock prices and other variables into the data variable by reading the .csv file
data = pd.read_csv('AMD.csv')
#Check to see the data is pulled correctly using the head() function

#Moving Average Model
def MovingAvg(mdata):
    closing_price = mdata["Adj Close"]
    date = mdata["Date"]
    pred_val = pd.DataFrame({'Date' : date,'Adj Close' : closing_price})

    for i in range(round(len(data)*0.7), len(data)):
        pred_val.at[i, 'Predicted Adj Close'] = sum(closing_price[i-20:i-1])/len(closing_price[i-20:i-1])
    #print(pred_val)

    plt.plot(date, closing_price, label = 'Real Prices')
    plt.plot(date, pred_val['Predicted Adj Close'], label = 'Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    #Calculate the error of the model
    err = 0
    for i in range(round(len(data)*0.7), len(data)):
        err = err + pow((pred_val.at[i,'Adj Close'] - pred_val.at[i,'Predicted Adj Close']),2)
    n = len(data)-round(len(data)*0.7)
    err = err/n
    print(err)

MovingAvg(data)     


