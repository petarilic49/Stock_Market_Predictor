import numpy as  np
import pandas as pd 
import sklearn.metrics as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from math import sqrt
from matplotlib import pyplot as plt
#K-Nearest Neighbour Method
#Once again, the opening price and volume will be assumed to be the independent variables of the model 
def Knearest(tdata):
    #Load the Opening Price, Volume, and Closing Price per day into a new Dataframe
    new_data = pd.DataFrame({'Open' : tdata['Open'], 'Volume' : tdata['Volume'], 'Close': tdata['Close']})
    
    #Split the Training and the Test data
    #Dont need to use the built in test_train_split because that will randomly pick points of our data but we need to use the first 70% of data for train
    X_train = pd.DataFrame({'Open': new_data['Open'][:round(len(new_data)*0.70)], 'Volume': new_data['Volume'][:round(len(new_data)*0.70)]})
    #Convert the DataFrame to a 2D array
    X_train = np.array(X_train[['Open','Volume']])
    Y_train = np.array(new_data['Close'][:round(len(new_data)*0.70)])

    #Initiate the testing data, which will be the last 30% of the data
    X_test = pd.DataFrame({'Open': new_data['Open'][round(len(new_data)*0.70):len(new_data)], 'Volume': new_data['Volume'][round(len(new_data)*0.70):len(new_data)]})
    #Convert the DataFrame to a 2D array
    X_test = np.array(X_test[['Open','Volume']])
    Y_test = np.array(new_data['Close'][round(len(new_data)*0.70):len(new_data)])

    #Need to scale the X data so that we dont have the volume scew the data 
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #Need to find the optimal K value. This can be done by evaluating the root mean square error for a series of K values chosen
    rms_vals = []
    for K in range(20):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors=K)
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)

        err = sqrt(sm.mean_squared_error(Y_test, prediction))
        rms_vals.append(err)
        print('The RMSE for K= ',K,'is: ',err)

    #We want K value with minimum RMSE error value which is approximately 9 or 2 according to the plot created below
    plt.plot(rms_vals)
    plt.title('Root Mean Square Error For Each K Value')
    plt.xlabel('K Value')
    plt.ylabel('Root Mean Square Error')
    plt.show()

    #There is a better way to find this out which K value is best
    #Using the GridSearchCV function we can loop through the predefined k_vals and fit the knn model on the training set
    #This will automatically take the best (ie lowest error) k value as for the KNN model 
    k_vals = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

    knn = neighbors.KNeighborsRegressor()

    model = GridSearchCV(knn, k_vals, cv=5)
    #Fit the KNN model with the training dataset
    model.fit(X_train,Y_train)
    #Print the chosen K value for the KNN model
    print(model.best_params_)
    
    #Predict the future closing prices based on the test dataset 
    prediction = model.predict(X_test)
    
    #For loop to populate the predicted closing price into the original Dataframe
    for i in range(round(len(new_data)*0.7), len(new_data)):
        new_data.at[i,'Predicted Close'] = prediction[i-176]

    #Plot the original as well as the predicted KNN model stock prices
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], new_data['Close'][round(len(new_data)*0.7):len(new_data)], label = 'Real Prices')
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], new_data['Predicted Close'][round(len(new_data)*0.7):len(new_data)], label = 'KNN Predicted Prices')
    plt.title('Real Closing Price vs KNN Model Predicted Closing Price')
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