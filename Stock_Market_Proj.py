#PURPOSE: to investigate the different machine learning regression algorithms and see which is best for predicting stock market prices
import numpy as  np
import pandas as pd 
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as sm
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
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
#However if we only use the date as an independent variable we will get very inaccurate future predictions which will not grasp the purpose of this project
#which is to learn how different ML regression algorithms work and how they can be used
#For this reason I am going to assume that the opening price and volume of the stock at the given day are our independent variables. This is because opening
#prices and volume are given at the opening of the stock market therefore our algorithm could predict the closing price based on these numbers for that day 

# def switch_func(d):
#     return{
#         'Monday': 1,
#         'Tuesday': 0,
#         'Wednesday': 0,
#         'Thursday': 0,
#         'Friday': 1
#     }.get(d, 0)

def LinearReg(tdata):
    #Load the Opening Price, Volume, and Closing Price per day into a new Dataframe
    new_data = pd.DataFrame({'Open' : tdata['Open'], 'Volume' : tdata['Volume'], 'Close': tdata['Close']})
    
    #Split the Training and the Test data
    #Dont need to use the built in test_train_split because that will randomly pick points of our data but we need to use the first 70% of data for train
    X_train = pd.DataFrame({'Open': new_data['Open'][:round(len(new_data)*0.70)], 'Volume': new_data['Volume'][:round(len(new_data)*0.70)]})
    #Convert the DataFrame to a 2D array
    X_train = np.array(X_train[['Open','Volume']])
    Y_train = np.array(new_data['Close'][:round(len(new_data)*0.70)])

    X_test = pd.DataFrame({'Open': new_data['Open'][round(len(new_data)*0.70):len(new_data)], 'Volume': new_data['Volume'][round(len(new_data)*0.70):len(new_data)]})
    #Convert the DataFrame to a 2D array
    X_test = np.array(X_test[['Open','Volume']])
    Y_test = np.array(new_data['Close'][round(len(new_data)*0.70):len(new_data)])

    #Need to create the matrices: X = [1 X1 X2] where X1 and X2 are column vectors of the open price and volume for the training data
    #We have to scale the values so there is no scews
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    col1 = np.ones((len(X_train),1))
    col2 = pd.DataFrame({'Open': new_data['Open'][:round(len(new_data)*0.70)]}).to_numpy()
    col2 = scaler.fit_transform(col2)
    col3 = pd.DataFrame({'Volume': new_data['Volume'][:round(len(new_data)*0.70)]}).to_numpy()
    col3 = scaler.fit_transform(col3)
    
    #Combine all three columns into a matrix, the hstack function only takes in one argument
    X_matrix = np.hstack((col1, col2, col3))
    Y_matrix = Y_train[:, np.newaxis]
    
    #Need to perform matrix algebra to get B values: B_vals = (X_transpose * X)^-1 * X_transpose*Y
    X_transpose = X_matrix.transpose()
    #Matrix multiplication is done with .dot() function in python since * operand would just perform element multiplication
    firstop = np.linalg.inv(X_transpose.dot(X_matrix))
    secondop = X_transpose.dot(Y_matrix)
    B_vals = firstop.dot(secondop)

    #The Linear Regression model looks as follows: Y_hat = B_not + B_1*X1 + B_2*X2
    for i in range(round(len(new_data)*0.7), len(new_data)):
        new_data.at[i,'Predicted Close'] = B_vals[0] + (B_vals[1]*X_test[i-176][0]) + (B_vals[2]*X_test[i-176][1])


    #Use Built in scikit Linear Regression model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)

    for i in range(round(len(new_data)*0.7), len(new_data)):
         new_data.at[i,'Predicted Close Model'] = prediction[i-176]

    plt.plot(tdata['Date'], new_data['Close'], label = 'Real Prices')
    plt.plot(tdata['Date'], new_data['Predicted Close'], label = 'Predicted Prices')
    plt.plot(tdata['Date'], new_data['Predicted Close Model'], label = 'Predicted Model Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    # Calculate the error of the model. Since both the created and the scikit models output the same value we will just use the scikit model to find accuracy
    print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, prediction), 2)) 
    print("Mean squared error =", round(sm.mean_squared_error(Y_test, prediction), 2)) 
    print("Median absolute error =", round(sm.median_absolute_error(Y_test, prediction), 2)) 
    print("Explain variance score =", round(sm.explained_variance_score(Y_test, prediction), 2)) 
    print("R2 score =", round(sm.r2_score(Y_test, prediction), 2))

#K-Nearest Neighbour Method
# def Knearest(tdata):
#     new_data = pd.DataFrame({'Date' : pd.to_datetime(tdata['Date']), 'Adj Close' : tdata['Adj Close']})
#     new_data['DayofWeek'] = new_data['Date'].dt.day_name()
#     for i in range(0, len(new_data)):
#         new_data.at[i, 'DayScale'] = switch_func(new_data['DayofWeek'][i])

#     #Dont need to scale it because the X_train and X_test are already between 0 and 1 and it only has one variable contributing to finding price
#     X_train = new_data['DayScale'][:round(len(new_data)*0.70)]
#     Y_train = new_data['Adj Close'][:round(len(new_data)*0.70)]
#     X_test = new_data['DayScale'][round(len(new_data)*0.70):len(new_data)]
#     Y_test = new_data['Adj Close'][round(len(new_data)*0.70):len(new_data)]

#     scaler = MinMaxScaler(feature_range=(0,1))
#     x_train = np.array(X_train)
#     x_train = x_train[:,np.newaxis]
#     x_test = np.array(X_test)
#     x_test = x_test[:,np.newaxis]
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.fit_transform(x_test)

#     k_vals = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}

#     knn = neighbors.KNeighborsRegressor()

#     model = GridSearchCV(knn, k_vals, cv=5)
#     model.fit(x_train_scaled,Y_train)
    
#     prediction = model.predict(x_test_scaled)
    
#     for i in range(round(len(new_data)*0.7), len(new_data)):
#         new_data.at[i,'Predicted Close'] = prediction[i-176]

#     plt.plot(new_data['Date'], new_data['Adj Close'], label = 'Real Prices')
#     plt.plot(new_data['Date'], new_data['Predicted Close'], label = 'Predicted Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Closing Price')
#     plt.legend()
#     plt.show()


#MovingAvg(data)     
LinearReg(data)
#Knearest(data)

