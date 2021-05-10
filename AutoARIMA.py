import numpy as  np
import pandas as pd 
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima


#Auto ARIMA Method
def AutoARIMA(tdata):
    #Load the data into a new variable
    #Since we are only looking at previous close prices, this is a univariate model
    #Take the close price of the first 70% for the training data, and the last 30% for the testing data
    #Both training and testing should be converted to arrays
    new_data = pd.DataFrame({'Close': tdata['Close']})
    Y_train = np.array(new_data['Close'][:round(len(new_data)*0.70)])
    Y_train = Y_train[:, np.newaxis]
    Y_test = np.array(new_data['Close'][round(len(new_data)*0.70):len(new_data)])
    Y_test = Y_test[:, np.newaxis]

    #We need to check if the time series is stationary. Just by looking at the stock prices we know its not but for good practice code should be implemented
    #Will use the augmented dickey fuller test to check if the data is stationary 
    stat_test = adfuller(Y_train, autolag="AIC")
    print(stat_test[1]) #The test shows that our p value is 0.723 which is much greater than 0.05. This means that we have 0.723 chance 
    #probability that null hypothesis will not be rejected and therefore we know our time series is NOT stationary

    #Now need to make our time series stationary
    #Will use the simplest method which is just differencing the time series (ie subtract the current value by the previous)
    new_data_stationary = new_data['Close'][:round(len(new_data)*0.70)].diff().dropna()
    Y_train_st = np.array(new_data_stationary)
    Y_train_st = Y_train[:, np.newaxis]

    #Check again if the time series is stationary
    stat_test = adfuller(Y_train_st, autolag="AIC")
    print(stat_test[1]) # The p value is much smaller than 0.05 therefore we can reject the Null Hypothesis and therefore the time series is now stationary
    #Plot the stationarized time series
    plt.plot(new_data_stationary, label = 'Difference')
    plt.title('Differenced Dataset (Stationary Dataset)')
    plt.xlabel('Time')
    plt.ylabel('Diff')
    plt.legend()
    plt.show()

    #Need to find the p and q values for the ARIMA model. We can do this by ploting the ACF and PACF 
    #To learn the theory of the model I will create the plots and try to pick appropriate p and q values, but in reality there is an auto_arima function that
    #does so automatically 
    #Rule for Choosing p: If the PACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is positive--i.e., if the series 
    #appears slightly "underdifferenced"--then consider adding an AR term to the model. The lag at which the PACF cuts off is the indicated number of AR terms.
    #Rule for Choosing q: If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series 
    #appears slightly "overdifferenced"--then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms.
    #Use the following link as reference: http://people.duke.edu/~rnau/411arim3.htm

    #Plot the ACF
    plot_acf(new_data_stationary)
    plt.show() #q = 1

    #Plot the PACF
    plot_pacf(new_data_stationary)
    plt.show() #p = 0

    #Implement the ARIMA model. Will use the auto_arima function that will automatically determine which type of ARIMA model fits best as well as determine 
    #the p, q, and d values. This can be compared to the values we saw from the graph
    auto_model = auto_arima(Y_train, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, d=None, seasonal=False) 
    print(auto_model.summary()) # Tells us they should be as follows: p = 0, d = 1, q = 1 
    start = len(Y_train)
    end = len(Y_train) + len(Y_test)
    #Predict the future close price with the auto generated ARIMA model above. Specified the time steps (ie periods) in the future we want to predict
    prediction = auto_model.predict(n_periods=(end-start))
    
    #For loop to populate the predicted closing price into the original Dataframe
    for i in range(round(len(new_data)*0.7), len(new_data)):
        new_data.at[i,'Predicted Close'] = prediction[i-176]
    
    #Plot the original as well as the predicted KNN model stock prices
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], new_data['Close'][round(len(new_data)*0.7):len(new_data)], label = 'Real Prices')
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], new_data['Predicted Close'][round(len(new_data)*0.7):len(new_data)], label = 'ARIMA Predicted Prices')
    plt.title('Real Closing Price vs ARIMA Predicted CLosing Price')
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