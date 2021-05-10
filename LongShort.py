import numpy as  np
import pandas as pd 
import sklearn.metrics as sm
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

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
    #Xtrain will be a 3D array which will hold rows of 35 previous time steps (closing prices) for each predicted closing price at time step t
    Xtrain = []
    #Ytrain will hold the actual closing price at time step t which will be compared to the predicted closing price from the LSTM network for losses and optimized
    Ytrain = []

    for i in range(35, len(X_train)):
        Xtrain.append(X_train[i-35:i])
        Ytrain.append(X_train[i])
    
    #Make the Xtrain and Ytrain an array
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    #Need to reshape the data for the LSTM model
    #First dimension is number of rows in tarining dataset, second dimension is the number of previous time steps that are used to predict current time step
    #Third dimension is the number of indicators which is just 1 because we are only using previous closing price to predict future closing price
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1],1))

    #Create the actual LSTM model
    #Sequential initializes the model to be a plain stack of layers where each layer has exactly one input tensor and one output tensor. This is important
    #When creating neural networks
    model = Sequential()
    #Add a LSTM layer to the neural network. The units is the number of neurons in the layer which was chosen to be 50 through trial and error
    #The return_sequence is set to True since we will add another LSTM layer thus we want a sequence to be outputed
    #Need to set an input_shape since its first layer, the input will be the number of previous time steps and the indicator will be 1 since just using closing price
    model.add(LSTM(units=50, return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
    #Dropout is initialized after every layer to ensure that our LSTM model does not undergo overfitting while training. An arbitrary value of 15% dropout was chosen
    model.add(Dropout(0.15))
    #Create another LSTM layer. The units is initialized to be the same again, however return_sequence is set to False. This is because we dont want an output
    #of the sequence but an output of just one value which is the predicted future price
    model.add(LSTM(units=50, return_sequences=False))
    #Dropout of 15% is initiailized again
    model.add(Dropout(0.15))
    #Add a dense layer at the end of our model with units value of 1. This is because we want to predict a single value in the output
    model.add(Dense(units=1))
    #Need to compile the above LSTM layers and initialize the way of calculating the loss when training the network, as well as what optimizer should be used
    #when back propagating to adjust the weightes of all the input parameters
    #Through research the Adam (AdaGrad and RMSProp algorithms) optimizer is shown to be best practise for many LSTM networks
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Fit the compiled LSTM network with the training data and train the neural network
    #An epoch size of 100 was chosen and a batch size of 16 was selected. This was experimented with however an epoch of 100 completes the program in optimal time 
    #Batch size is fairly low since we only have 252 data entries in total and we are only looking at the previous 35 to predict the future closing price
    model.fit(Xtrain, Ytrain, epochs=100, batch_size=16)

    #Test the LSTM model with the test data 
    #First we need to get the testing data 
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 35:].values
    #Need to scale the testing data 
    inputs = scaler.fit_transform(inputs)

    #Populate the testing data by creating a 3D array that starts with the previous 30 time steps from the end of the training data and so forth
    test_data = []
    for i in range(35, len(inputs)):
        test_data.append(inputs[i-35:i])
    test_data = np.array(test_data)
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1],1))

    #Make the predictions
    predictions = model.predict(test_data)
    #The predicted closing stiock prices are scaled, therefore convert them back to the unscaled values
    predictions = scaler.inverse_transform(predictions)

    #For loop to populate the predicted closing price into the original Dataframe
    for i in range(round(len(dataset_total)*0.7), len(dataset_total)):
        dataset_total.at[i,'Predicted Close'] = predictions[i-176]
    
    #Plot the predicted closing stock price with the actual close price of the test data
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], dataset_total['Close'][round(len(dataset_total)*0.7):len(dataset_total)], label = 'Real Prices')
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], dataset_total['Predicted Close'][round(len(dataset_total)*0.7):len(dataset_total)], label = 'LSTM Predicted Prices')
    plt.title('Real Closing Price vs LSTM Model Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    # Calculate the error of the model
    # Gather the actual closing price and set to Y_test
    prediction = dataset_total['Predicted Close'][round(len(dataset_total)*0.7):len(dataset_total)]
    print("Mean absolute error =", round(sm.mean_absolute_error(dataset_test, prediction), 2)) 
    print("Mean squared error =", round(sm.mean_squared_error(dataset_test, prediction), 2)) 
    print("Median absolute error =", round(sm.median_absolute_error(dataset_test, prediction), 2)) 
    print("Explain variance score =", round(sm.explained_variance_score(dataset_test, prediction), 2)) 
    print("R2 score =", round(sm.r2_score(dataset_test, prediction), 2))