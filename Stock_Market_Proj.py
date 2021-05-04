#PURPOSE: to investigate the different machine learning regression algorithms and see which is best for predicting stock market prices
import numpy as  np
import pandas as pd 
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#We will use AMD stock prices for example
#Load the stock prices and other variables into the data variable by reading the .csv file
data = pd.read_csv('AMD.csv')
#Check to see the data is pulled correctly using the head() function

#Moving Average Model
# def MovingAvg(mdata):
#     closing_price = mdata["Adj Close"]
#     date = mdata["Date"]
#     pred_val = pd.DataFrame({'Date' : date,'Adj Close' : closing_price})

#     for i in range(round(len(data)*0.7), len(data)):
#         pred_val.at[i, 'Predicted Adj Close'] = sum(closing_price[i-20:i-1])/len(closing_price[i-20:i-1])
#     #print(pred_val)

#     plt.plot(date, closing_price, label = 'Real Prices')
#     plt.plot(date, pred_val['Predicted Adj Close'], label = 'Predicted Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Closing Price')
#     plt.legend()
#     plt.show()

#     #Calculate the error of the model
#     err = 0
#     for i in range(round(len(data)*0.7), len(data)):
#         err = err + pow((pred_val.at[i,'Adj Close'] - pred_val.at[i,'Predicted Adj Close']),2)
#     n = len(data)-round(len(data)*0.7)
#     err = err/n
#     print(err)

#Linear Regression Model
#I have coded my own linear regression algorithm, as well as tested it with the scikit learn built in linear regression
#Thinking practically the only independent variable we have in this application is the date since we will not know the opening, high, low, and volume 
#of the given stock for the future, we only know the date
#According to Investopedia, the 'Weekend Effect' states the average return on Fridays usually exceeds the average return on Mondays 
#Therefore the regression algorithm will favor Friday's as a sign of increasing close prices

def switch_func(d):
    return{
        'Monday': 0.2,
        'Tuesday': 0.4,
        'Wednesday': 0.6,
        'Thursday': 0.8,
        'Friday': 1
    }.get(d, 0)

def LinearReg(tdata):
    new_data = pd.DataFrame({'Date' : pd.to_datetime(tdata['Date']), 'Adj Close' : tdata['Adj Close']})
    new_data['DayofWeek'] = new_data['Date'].dt.day_name()
    for i in range(0, len(new_data)):
        new_data.at[i, 'DayScale'] = switch_func(new_data['DayofWeek'][i])

    X_train = new_data['DayScale'][:round(len(new_data)*0.70)]
    Y_train = new_data['Adj Close'][:round(len(new_data)*0.70)]
    X_test = new_data['DayScale'][round(len(new_data)*0.70):len(new_data)]
    Y_test = new_data['Adj Close'][round(len(new_data)*0.70):len(new_data)]

    #Y_hat = B_not + B_1*X
    Y_mean = 0
    X_mean = 0
    for i in range (0, len(Y_train)):
        Y_mean = Y_mean + Y_train.at[i]
        X_mean = X_mean + X_train.at[i]
    Y_mean = Y_mean / len(Y_train)
    X_mean = X_mean / len(X_train)

    #Calculating B_1
    B_num = 0
    B_den = 0
    for i in range(0,len(X_train)):
        B_num = B_num + (X_train.at[i]*Y_train.at[i] - Y_mean*X_train.at[i])
        B_den = B_den + (pow(X_train.at[i],2) - X_mean*X_train.at[i])
    B_one = B_num / B_den
    

    #Calculating B_not
    B_not = Y_mean - B_one*X_mean
    
    #Create the Linear Regression Model
    for i in range(round(len(new_data)*0.7), len(new_data)):
        new_data.at[i,'Predicted Close'] = B_not + B_not*new_data.at[i,'DayScale']
    

    model = LinearRegression()
    xtrain = np.array(X_train)
    xtrain = xtrain[:, np.newaxis]
    xtest = np.array(X_test)
    xtest = xtest[:, np.newaxis]

    model.fit(xtrain, Y_train)
    prediction = model.predict(xtest)
    #print(prediction)

    for i in range(round(len(new_data)*0.7), len(new_data)):
        new_data.at[i,'Predicted Close Model'] = prediction[i-176]

    plt.plot(new_data['Date'], new_data['Adj Close'], label = 'Real Prices')
    plt.plot(new_data['Date'], new_data['Predicted Close'], label = 'Predicted Prices')
    plt.plot(new_data['Date'], new_data['Predicted Close Model'], label = 'Predicted Model Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()
  

    #Calculate the error of the model
    # err = 0
    # for i in range(round(len(new_data)*0.7), len(new_data)):
    #     err = err + pow((new_data.at[i,'Adj Close'] - new_data.at[i,'Predicted Close']),2)
    # n = len(new_data)-round(len(new_data)*0.7)
    # err = err/n
    # print(err)


#K-Nearest Neighbour Method

#MovingAvg(data)     
LinearReg(data)

