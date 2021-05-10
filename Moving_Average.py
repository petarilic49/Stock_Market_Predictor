#Import all the necessary libraries 
import numpy as  np
import pandas as pd 
import sklearn.metrics as sm
from math import sqrt
from matplotlib import pyplot as plt

#Moving Average Model
def MovingAvg(tdata):
    #Load the closing price and date into tow seperate variables
    closing_price = tdata["Close"]
    date = tdata["Date"]
    #Combine the two variables holding the closing price and date into one Dataframe series. This will be used later for plotting
    pred_val = pd.DataFrame({'Date' : date,'Close' : closing_price})

    #The predicted future price will be calculated by the average of the 20 previous prices. Assuming we want to predict the last 30% of the data
    #For loop is used to iterate through the last 30% of data points and take the moving average of the last 20 data points for each time step (ie day) 
    for i in range(round(len(tdata)*0.7), len(tdata)):
        pred_val.at[i, 'Predicted Close'] = sum(closing_price[i-20:i-1])/len(closing_price[i-20:i-1])
    

    #Plot the original as well as the predicted moving average stock prices
    plt.plot(date[round(len(date)*0.7):len(date)], closing_price[round(len(closing_price)*0.7):len(closing_price)], label = 'Real Prices')
    plt.plot(date[round(len(date)*0.7):len(date)], pred_val['Predicted Close'][round(len(pred_val)*0.7):len(pred_val)], label = 'Moving Average Predicted Prices')
    plt.title('Real Closing Price vs Moving Average Model Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    # Calculate the error of the model
    # Gather the actual closing price and set to Y_test
    Y_test = closing_price[round(len(closing_price)*0.7):len(closing_price)]
    prediction = pred_val['Predicted Close'][round(len(pred_val)*0.7):len(pred_val)]
    print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, prediction), 2)) 
    print("Mean squared error =", round(sm.mean_squared_error(Y_test, prediction), 2)) 
    print("Median absolute error =", round(sm.median_absolute_error(Y_test, prediction), 2)) 
    print("Explain variance score =", round(sm.explained_variance_score(Y_test, prediction), 2)) 
    print("R2 score =", round(sm.r2_score(Y_test, prediction), 2))