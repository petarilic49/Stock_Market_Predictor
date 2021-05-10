import numpy as  np
import pandas as pd 
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from matplotlib import pyplot as plt

#Linear Regression Model
#I have decided to code my own linear regression algorithm, as well as tested it with the scikit learn built in linear regression
#Thinking practically the only independent variable we have in this application is the date since we will not know the opening, high, low, and volume 
#of the given stock for the future
#However if we only use the date as an independent variable we will get very inaccurate future predictions which will not grasp the purpose of this project
#which is to learn how different Regression Machine Learning algorithms work and how they can be implemented 
#For this reason I am going to assume that the opening price and volume of the stock at the given day are our independent variables. This is because opening
#prices and volume are given at the opening of the stock market therefore our algorithm could predict the closing price based on these numbers for that day 

def LinearReg(tdata):
    #Load the Opening Price, Volume, and Closing Price per day into a new Dataframe
    new_data = pd.DataFrame({'Open' : tdata['Open'], 'Volume' : tdata['Volume'], 'Close': tdata['Close']})
    
    #Split the Training and the Test data
    #Dont need to use the built in test_train_split because that will randomly pick points of our data but we need to use the first 70% of data for train
    X_train = pd.DataFrame({'Open': new_data['Open'][:round(len(new_data)*0.70)], 'Volume': new_data['Volume'][:round(len(new_data)*0.70)]})
    #Convert the training DataFrame to a 2D array
    X_train = np.array(X_train[['Open','Volume']])
    Y_train = np.array(new_data['Close'][:round(len(new_data)*0.70)])

    #Initiate the testing data, which will be the last 30% of the data 
    X_test = pd.DataFrame({'Open': new_data['Open'][round(len(new_data)*0.70):len(new_data)], 'Volume': new_data['Volume'][round(len(new_data)*0.70):len(new_data)]})
    #Convert the testing DataFrame to a 2D array
    X_test = np.array(X_test[['Open','Volume']])
    Y_test = np.array(new_data['Close'][round(len(new_data)*0.70):len(new_data)])

    #THE FOLLOWING CODE BELOW IS THE DERIVED LINEAR REGRESSION MODEL WITHOUT SCIKIT LEARN
    #Since we are dealing with a multivariable linear regression problem, it makes most sense to deal with matrix algebra for cleanliness
    #The Linear Regression Model we are trying to solve for is as follows: Y_hat = B_not + B_1*X1 + B_2*X2, where B values are constants (ie weights)
    #while X1 and X2 refer to our independent variables which are the opening price and volume
    
    #Before we start it is important to scale the data especially since we are dealing with two highly variant variables
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    #To solve for the B values we need to create the matrix: X = [1 X1 X2] where X1 and X2 are column vectors of the open price and volume for the training data
    #We have to scale the values so there is no scews
    col1 = np.ones((len(X_train),1))
    col2 = []
    col3 = []
    #Use a for loop to iterate through training data and populate col2 with opening price and col3 with volume
    for i in range(len(X_train)):
        col2.append(X_train[i][0])
        col3.append(X_train[i][1])
    #Convert col2 and col3 from lists to arrays and make them a column vector
    col2 = np.array(col2)
    col3 = np.array(col3)
    col2 = col2[:, np.newaxis]
    col3 = col3[:, np.newaxis]
    
    #Combine all three columns into a matrix, the hstack function only takes in one argument
    X_matrix = np.hstack((col1, col2, col3))
    #Y_matrix will just be a column vector with the closing prices associated to each day of the training data
    Y_matrix = Y_train[:, np.newaxis]
    
    #Need to perform matrix algebra to get B values: B_vals = (X_transpose * X)^-1 * X_transpose*Y
    X_transpose = X_matrix.transpose()
    #Matrix multiplication is done with .dot() function in python since * operand would just perform element multiplication
    #firststop = (X_transpose * X)^-1, secondstop = X_transpose * Y
    firstop = np.linalg.inv(X_transpose.dot(X_matrix))
    secondop = X_transpose.dot(Y_matrix)
    B_vals = firstop.dot(secondop)

    #The Linear Regression model looks as follows: Y_hat = B_not + B_1*X1 + B_2*X2
    #For loop to populate the predicted closing price of the created linear regression model into the original Dataframe
    for i in range(round(len(new_data)*0.7), len(new_data)):
        new_data.at[i,'Predicted Close'] = B_vals[0] + (B_vals[1]*X_test[i-176][0]) + (B_vals[2]*X_test[i-176][1])

    #THE FOLLOWING CODE BELOW USES THE SCIKIT LEARN LIBRARY TO PERFORM LINEAR REGRESSION ON THE DATASET
    #Initialize the Linear Regression model from the scikit learn library
    model = LinearRegression()
    #Fit the training data into the model
    model.fit(X_train, Y_train)
    #Predict the future closing price using the testing dataset
    prediction = model.predict(X_test)

    #For loop to populate the predicted closing price of the linear regression model into the original Dataframe
    for i in range(round(len(new_data)*0.7), len(new_data)):
         new_data.at[i,'Predicted Close Model'] = prediction[i-176]

    #Plot both created and scikit learn Linear Regression model on the same plot 
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], new_data['Close'][round(len(new_data)*0.7):len(new_data)], label = 'Real Prices')
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], new_data['Predicted Close'][round(len(new_data)*0.7):len(new_data)], label = 'Linear Regression Predicted Prices (Derived Model)')
    plt.plot(tdata['Date'][round(len(tdata)*0.7):len(tdata)], new_data['Predicted Close Model'][round(len(new_data)*0.7):len(new_data)], label = 'Linear Regression Predicted Prices (SciKit Model)')
    plt.title('Real Closing Price vs Linear Regression Model Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    # Calculate the error of the model. Since both the created and the scikit model output the same value we will just use the scikit model to find accuracy
    print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, prediction), 2)) 
    print("Mean squared error =", round(sm.mean_squared_error(Y_test, prediction), 2)) 
    print("Median absolute error =", round(sm.median_absolute_error(Y_test, prediction), 2)) 
    print("Explain variance score =", round(sm.explained_variance_score(Y_test, prediction), 2)) 
    print("R2 score =", round(sm.r2_score(Y_test, prediction), 2))