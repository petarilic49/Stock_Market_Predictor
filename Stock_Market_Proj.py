#PURPOSE: to investigate the different machine learning regression algorithms and see which is best for predicting stock market prices
import numpy as  np
import pandas as pd 
# import sklearn.metrics as sm
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn import neighbors
# from sklearn.model_selection import GridSearchCV
from math import sqrt
from matplotlib import pyplot as plt
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima.model import ARIMA
# from pmdarima.arima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

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

# def LinearReg(tdata):
#     #Load the Opening Price, Volume, and Closing Price per day into a new Dataframe
#     new_data = pd.DataFrame({'Open' : tdata['Open'], 'Volume' : tdata['Volume'], 'Close': tdata['Close']})
    
#     #Split the Training and the Test data
#     #Dont need to use the built in test_train_split because that will randomly pick points of our data but we need to use the first 70% of data for train
#     X_train = pd.DataFrame({'Open': new_data['Open'][:round(len(new_data)*0.70)], 'Volume': new_data['Volume'][:round(len(new_data)*0.70)]})
#     #Convert the DataFrame to a 2D array
#     X_train = np.array(X_train[['Open','Volume']])
#     Y_train = np.array(new_data['Close'][:round(len(new_data)*0.70)])

#     X_test = pd.DataFrame({'Open': new_data['Open'][round(len(new_data)*0.70):len(new_data)], 'Volume': new_data['Volume'][round(len(new_data)*0.70):len(new_data)]})
#     #Convert the DataFrame to a 2D array
#     X_test = np.array(X_test[['Open','Volume']])
#     Y_test = np.array(new_data['Close'][round(len(new_data)*0.70):len(new_data)])

#     #Need to create the matrices: X = [1 X1 X2] where X1 and X2 are column vectors of the open price and volume for the training data
#     #We have to scale the values so there is no scews
#     scaler = MinMaxScaler(feature_range=(0,1))
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.fit_transform(X_test)
#     col1 = np.ones((len(X_train),1))
#     col2 = pd.DataFrame({'Open': new_data['Open'][:round(len(new_data)*0.70)]}).to_numpy()
#     col2 = scaler.fit_transform(col2)
#     col3 = pd.DataFrame({'Volume': new_data['Volume'][:round(len(new_data)*0.70)]}).to_numpy()
#     col3 = scaler.fit_transform(col3)
    
#     #Combine all three columns into a matrix, the hstack function only takes in one argument
#     X_matrix = np.hstack((col1, col2, col3))
#     Y_matrix = Y_train[:, np.newaxis]
    
#     #Need to perform matrix algebra to get B values: B_vals = (X_transpose * X)^-1 * X_transpose*Y
#     X_transpose = X_matrix.transpose()
#     #Matrix multiplication is done with .dot() function in python since * operand would just perform element multiplication
#     firstop = np.linalg.inv(X_transpose.dot(X_matrix))
#     secondop = X_transpose.dot(Y_matrix)
#     B_vals = firstop.dot(secondop)

#     #The Linear Regression model looks as follows: Y_hat = B_not + B_1*X1 + B_2*X2
#     for i in range(round(len(new_data)*0.7), len(new_data)):
#         new_data.at[i,'Predicted Close'] = B_vals[0] + (B_vals[1]*X_test[i-176][0]) + (B_vals[2]*X_test[i-176][1])


#     #Use Built in scikit Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, Y_train)
#     prediction = model.predict(X_test)

#     for i in range(round(len(new_data)*0.7), len(new_data)):
#          new_data.at[i,'Predicted Close Model'] = prediction[i-176]

#     plt.plot(tdata['Date'], new_data['Close'], label = 'Real Prices')
#     plt.plot(tdata['Date'], new_data['Predicted Close'], label = 'Predicted Prices')
#     plt.plot(tdata['Date'], new_data['Predicted Close Model'], label = 'Predicted Model Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Closing Price')
#     plt.legend()
#     plt.show()

#     # Calculate the error of the model. Since both the created and the scikit models output the same value we will just use the scikit model to find accuracy
#     print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, prediction), 2)) 
#     print("Mean squared error =", round(sm.mean_squared_error(Y_test, prediction), 2)) 
#     print("Median absolute error =", round(sm.median_absolute_error(Y_test, prediction), 2)) 
#     print("Explain variance score =", round(sm.explained_variance_score(Y_test, prediction), 2)) 
#     print("R2 score =", round(sm.r2_score(Y_test, prediction), 2))

#K-Nearest Neighbour Method
#The same independent variable assumptions will be made for KNN method 
# def Knearest(tdata):
#     #Load the Opening Price, Volume, and Closing Price per day into a new Dataframe
#     new_data = pd.DataFrame({'Open' : tdata['Open'], 'Volume' : tdata['Volume'], 'Close': tdata['Close']})
    
#     #Split the Training and the Test data
#     #Dont need to use the built in test_train_split because that will randomly pick points of our data but we need to use the first 70% of data for train
#     X_train = pd.DataFrame({'Open': new_data['Open'][:round(len(new_data)*0.70)], 'Volume': new_data['Volume'][:round(len(new_data)*0.70)]})
#     #Convert the DataFrame to a 2D array
#     X_train = np.array(X_train[['Open','Volume']])
#     Y_train = np.array(new_data['Close'][:round(len(new_data)*0.70)])

#     X_test = pd.DataFrame({'Open': new_data['Open'][round(len(new_data)*0.70):len(new_data)], 'Volume': new_data['Volume'][round(len(new_data)*0.70):len(new_data)]})
#     #Convert the DataFrame to a 2D array
#     X_test = np.array(X_test[['Open','Volume']])
#     Y_test = np.array(new_data['Close'][round(len(new_data)*0.70):len(new_data)])

#     #Need to scale the X data so that we dont have the volume scew the data 
#     scaler = MinMaxScaler(feature_range=(0,1))
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.fit_transform(X_test)

#     #Need to find the optimal K value. This can be done by evaluating the root mean square error for a series of K values chosen
#     # rms_vals = []
#     # for K in range(20):
#     #     K = K+1
#     #     model = neighbors.KNeighborsRegressor(n_neighbors=K)
#     #     model.fit(X_train, Y_train)
#     #     prediction = model.predict(X_test)

#     #     err = sqrt(sm.mean_squared_error(Y_test, prediction))
#     #     rms_vals.append(err)
#     #     print('The RMSE for K= ',K,'is: ',err)

#     # #We want K value with minimum RMSE error value which is approximately 9 or 2 according to the plot created below. There is a better way to find this out
#     # plt.plot(rms_vals)
#     # plt.show()

#     k_vals = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

#     knn = neighbors.KNeighborsRegressor()

#     model = GridSearchCV(knn, k_vals, cv=5)
#     model.fit(X_train,Y_train)
#     print(model.best_params_)
    
#     prediction = model.predict(X_test)
    
#     for i in range(round(len(new_data)*0.7), len(new_data)):
#         new_data.at[i,'Predicted Close'] = prediction[i-176]

#     plt.plot(tdata['Date'], new_data['Close'], label = 'Real Prices')
#     plt.plot(tdata['Date'], new_data['Predicted Close'], label = 'Predicted Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Closing Price')
#     plt.legend()
#     plt.show()

#     # Calculate the error of the model. Since both the created and the scikit models output the same value we will just use the scikit model to find accuracy
#     print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, prediction), 2)) 
#     print("Mean squared error =", round(sm.mean_squared_error(Y_test, prediction), 2)) 
#     print("Median absolute error =", round(sm.median_absolute_error(Y_test, prediction), 2)) 
#     print("Explain variance score =", round(sm.explained_variance_score(Y_test, prediction), 2)) 
#     print("R2 score =", round(sm.r2_score(Y_test, prediction), 2))

#Auto ARIMA Method
# def AutoARIMA(tdata):
#     #Load the data into a new variable
#     #Since we are only looking at previous close prices, this is a univariate model itself
#     new_data = pd.DataFrame({'Close': tdata['Close']})
#     Y_train = np.array(new_data['Close'][:round(len(new_data)*0.70)])
#     Y_train = Y_train[:, np.newaxis]
#     Y_test = np.array(new_data['Close'][round(len(new_data)*0.70):len(new_data)])
#     Y_test = Y_test[:, np.newaxis]


#     #We need to check if the time series is stationary. Just by looking at the stock prices we know its not but for good practice I'll implement the code
#     #I will use the augmented dickey fuller test to check if the data is stationary 
#     stat_test = adfuller(Y_train, autolag="AIC")
#     #print(stat_test[1]) #The test shows that our p value is 0.723 which is much greater than 0.05. This means that we have 0.723 chance 
#     #probability that null hypothesis will not be rejected and therefore we know our time series is NOT stationary

#     #We now need to make our time series stationary
#     #We will use the simplest method which is just differencing the time series (ie subtract the current value by the previous)
#     new_data_stationary = new_data['Close'][:round(len(new_data)*0.70)].diff().dropna()
#     Y_train_st = np.array(new_data_stationary)
#     Y_train_st = Y_train[:, np.newaxis]

#     #Check again if the time series is stationary
#     stat_test = adfuller(Y_train_st, autolag="AIC")
#     #print(stat_test[1]) # The p value is much smaller than 0.05 therefore we can reject the Null Hypothesis and therefore the time series is now stationary
#     #Plot the stationarized time series
#     # plt.plot(new_data_stationary, label = 'Difference')
#     # plt.xlabel('Time')
#     # plt.ylabel('Diff')
#     # plt.legend()
#     # plt.show()

#     #Need to find the p and q values for the ARIMA model. We can do this by ploting the ACF and PACF 
#     #To learn the theory of the model I will create the plots and try to pick appropriate p and q values, but in reality there is an auto_arima function that would do 
#     #so automatically 
#     #Rule for Choosing p: If the PACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is positive--i.e., if the series 
#     #appears slightly "underdifferenced"--then consider adding an AR term to the model. The lag at which the PACF cuts off is the indicated number of AR terms.
#     #Rule for Choosing q: If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series 
#     #appears slightly "overdifferenced"--then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms.
#     #Use the following link as reference: http://people.duke.edu/~rnau/411arim3.htm

#     #Plot the ACF
#     #plot_acf(Y_train)
#     #plt.show() #q = 1

#     #Plot the PACF
#     #plot_pacf(Y_train)
#     #plt.show() #p = 0
#     auto_model = auto_arima(Y_train, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, d=None, seasonal=False) # See what auto_arima tells us the p, q, and d values should be
#     print(auto_model.summary()) # Tells us they should be as follows: p = 0, d = 1, q = 1 
#     start = len(Y_train)
#     end = len(Y_train) + len(Y_test)
#     prediction = auto_model.predict(n_periods=(end-start))
    
#     for i in range(round(len(new_data)*0.7), len(new_data)):
#         new_data.at[i,'Predicted Close'] = prediction[i-176]
    
#     plt.plot(tdata['Date'], new_data['Close'], label = 'Real Prices')
#     plt.plot(tdata['Date'], new_data['Predicted Close'], label = 'Predicted Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Closing Price')
#     plt.legend()
#     plt.show()

#     # Calculate the error of the model. Since both the created and the scikit models output the same value we will just use the scikit model to find accuracy
#     print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, prediction), 2)) 
#     print("Mean squared error =", round(sm.mean_squared_error(Y_test, prediction), 2)) 
#     print("Median absolute error =", round(sm.median_absolute_error(Y_test, prediction), 2)) 
#     print("Explain variance score =", round(sm.explained_variance_score(Y_test, prediction), 2)) 
#     print("R2 score =", round(sm.r2_score(Y_test, prediction), 2))
    

def LongShort(tdata):
     #Load the data into a new variable
    #Since we are only looking at previous close prices, this is a univariate model itself
    #It is also good idea to normalize the data for easier training by the LSTM 
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_total = pd.DataFrame({'Close': tdata['Close']})
    dataset_train = pd.DataFrame(dataset_total[:round(len(tdata)*0.70)])
    dataset_test = pd.DataFrame(dataset_total[round(len(tdata)*0.70):len(tdata)])
    #Convert to 1D array
    X_train = np.array(dataset_train)
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.flatten()

    #Split the training data into the appropriate form for LSTM model. LSTM expects data to be in a 3D array
    Xtrain = []
    Ytrain = []

    for i in range(35, len(X_train)):
        Xtrain.append(X_train[i-35:i])
        Ytrain.append(X_train[i])
    #Make the Xtrain and Ytrain an array
    Xtrain, Ytrain = np.array(Xtrain), np.array(Ytrain)

    #Need to reshape the data for the LSTM model
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1],1))

    #Create the actual LSTM model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
    model.add(Dropout(0.15))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.15))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(Xtrain, Ytrain, epochs=100, batch_size=16)

    #Test the LSTM model with the test data 
    #First we need to get the testing data 
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 35:].values
    inputs = scaler.fit_transform(inputs)

    test_data = []
    for i in range(35, len(inputs)):
        test_data.append(inputs[i-35:i])
    test_data = np.array(test_data)
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1],1))

    #Make the predictions
    predictions = model.predict(test_data)
    predictions = scaler.inverse_transform(predictions)
    print(len(predictions))

    for i in range(round(len(dataset_total)*0.7), len(dataset_total)):
        dataset_total.at[i,'Predicted Close'] = predictions[i-176]
    
    plt.plot(tdata['Date'], dataset_total['Close'], label = 'Real Prices')
    plt.plot(tdata['Date'], dataset_total['Predicted Close'], label = 'Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()


  


#MovingAvg(data)     
#LinearReg(data)
#Knearest(data)
#AutoARIMA(data)
LongShort(data)
