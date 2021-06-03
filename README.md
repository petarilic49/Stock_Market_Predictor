# Stock Market Forecast Project

# Introduction
The purpose of this project is to investigate five different regression machine learning algorithms and see which one is best for predicting future closing stock prices. The five algorithms chosen vary from begginner implementation difficulty to advanced implementation difficulty. The five algorithms tested are as such: 
  - Moving Average (Beginner)
  - Linear Regression
  - K Nearest Neighbor
  - Auto Regressive Integrated Moving Average (ARIMA)
  - Long Short Term Memory Recurrent Neural Network (Advanced)

To test the algorithms, one year historical stock data from Advanced Micro Devices (AMD) was taken from Yahoo Finance and downloaded into a .csv file. AMD was chosen arbitrarily and any other stock from Yahoo Finance could work with this code. The program begins by plotting the closing price of AMD over the past year and clearly illustrating which time steps will be used for training and which time steps will be used for testing. The plot below displays this clearly. 
![image](https://user-images.githubusercontent.com/37299986/117724558-edb01600-b1b1-11eb-943b-1e59e877dd7d.png)
Each algorithm will also be evaluated by their mean absolute error, mean squared error, median absolute error, explain variance score, and R2 score. 

# Moving Average
The moving average algorithm is the simplest of the five, as it only takes the average of the previous 20 time steps (ie 20 previous closing prices) to predict the future closing price of the stock. In order to do this a for loop was utilized to continuously input the averaged predicted future price of the stock. The plot below displays the real closing price versus the predicted moving average closing price of the stock. 
![image](https://user-images.githubusercontent.com/37299986/117724689-1f28e180-b1b2-11eb-9663-4067084885b9.png)
The moving average model contained the following accuracy scores: 
  - Mean absolute error = 3.32
  - Mean squared error = 17.03
  - Median absolute error = 2.89
  - Explain variance score = 0.47
  - R2 score = 0.36

# Linear Regression
Linear Regression is a statistical approach to modelling the relationship between a scalar repsonse and one or more independent variables. The Linear Regression equation can be modeled as Y_hat = B_not + B_1*X1 ... B_n*Xn, where the B values are constants or weights and the X values are the independent variables. Thinking practically the only independent variable we have in this application is the date since we will not know the opening, high, low, and volume of the given stock for the future. However if we only use the date as an independent variable we will get very inaccurate future predictions. For this reason its assumed that the opening price and volume of the stock at the given day are our independent variables. This is because opening prices and volume are given at the opening of the stock market therefore our algorithm could predict the closing price based on these numbers for that day. To ease program implementation, matrix algebra is used to solve for the B values using the training data. The predicted closing prices are computed by using the testing independent variables and the obtained Linear Regression model. 
The Linear Regression model was created without the scikit learn library to test the theory behind the algorithm. To check that the implemented model is correct, the scikit learn Linear Regression function was used to check the predicted closing prices match the created model. 
The following plot displays the predicted closing prices using the Linear Regression model versus the real closing price. 
![image](https://user-images.githubusercontent.com/37299986/117730464-5dc29a00-b1ba-11eb-83da-b3a84277ff16.png)
From the plot above it is evident that the created linear regression model and the scikit learn linear regression model output the same predicted prices which validates that the created linear regression model is correct. In addition, the linear regression model obtained the following accuracy scores: 
- Mean absolute error = 12.77
- Mean squared error = 223.58
- Median absolute error = 13.59
- Explain variance score = -1.7
- R2 score = -7.34

# K Nearest Neighbor
K Nearest Neighbor is a statistical algorithm which stores previously known cases to predict new cases based on similarity (ie where the nearest neighbor comes from). K Nearest Neighbor algorithms work for both classification and regression problems which makes it applicable for this application. In this situation, the training data was fitted into the scikit learn K Nearest Neighbor model where the opening price and volume were the classifiers for the closing price. Once the model was fitted, the testing datat was fed through the model where the model takes the average of the nearest neighbors (ie closest data points) to determine the closing stock price.
Determining the K value could be chosen graphically by using a GridSearchCV to test the Root Mean Square Error for a set of K values. In this situation a K value of 2 was chosen and is shown to have the lowest Root Mean Square Error as seen by the plot below. 
![image](https://user-images.githubusercontent.com/37299986/117736453-9b78f000-b1c5-11eb-8e3b-a4b113b6c5ed.png)
Once the K value is chosen the model was tested using the testing data. The plot below displays the K Nearest Neighbor predicted closing price versus the real closing price of the stock. 
![image](https://user-images.githubusercontent.com/37299986/117736525-d11dd900-b1c5-11eb-9d19-219e33c2e5d8.png)
The K Neareast Neighbor algorithm obtained the following accuracy scores: 
- Mean absolute error = 13.47
- Mean squared error = 271.89
- Median absolute error = 9.3
- Explain variance score = -2.57
- R2 score = -9.14

# Auto Regressive Integrated Moving Average (ARIMA)
ARIMA is a popular statistical algorithm which when given a stationary time series can forecast future values based on its own past values by analyzing its own lags and the lagged forecast errors. The most important aspect when dealing with ARIMA models is ensuring the time series data is stationary which is defined by having a constant mean over time, consant variance over time, and no seasonality. A popular test for determining if a time series is stationary is the Augmented Dickey Fuller test which was utilized in the coded ARIMA model. In terms of this specific application, stock price trends are almost always non-stationary therefore the training data was turned into a stationary dataset by implementing the difference method (ie difference current time step value by previous time step value). The plot below displays the stationary training data after differencing. 
![image](https://user-images.githubusercontent.com/37299986/117737229-6b325100-b1c7-11eb-88d6-42227f6f3982.png)
The ARIMA model contains three vital parameters which are the p (order of auto regression), d (number of differencing), and q (order of moving average). The p value can be obtained by observing a Partial Autocorelation Function Plot (PACF), while the q value can be obtained by observing a Autocorrelation Function Plot (ACF). The d value is the number of times the differencing method was applied (ie in our case just once therefore d=1). The PACF and ACF plots below were obtained for the stationary training dataset and imply a p value of 0 and a q value of 1. 
![image](https://user-images.githubusercontent.com/37299986/117737636-45597c00-b1c8-11eb-824c-a1bb6b6d3e66.png)
![image](https://user-images.githubusercontent.com/37299986/117737663-51453e00-b1c8-11eb-8977-1b1de3efd726.png)
Luckily in python, the pmdarima library contains an 'auto_arima' function which automatically caluclates which ARIMA model is most appropriate for the given time series, as well as automatically calculate which p, q, and d values are optimal for the chosen model. The plot below displays the predicted closing prices from the ARIMA model versus the real closing prices of the stock. 
![image](https://user-images.githubusercontent.com/37299986/117737826-abde9a00-b1c8-11eb-8788-c8bf3500e12b.png)
The ARIMA model obtained the following accuracy scores: 
- Mean absolute error = 20.96
- Mean squared error = 534.4
- Median absolute error = 24.41
- Explain variance score = -2.55
- R2 score = -18.94

# Long Short Term Memory Recurrent Neural Network
Long Short Term Memory Neural Networks are a type of Recurrent Neural Network that is capable of learning order dependences in a sequence prediction problem. Simple put, they solve the long term memory inefficiency that regular Neural Networks possess. This type of network takes in a sequence of previous data to predict future data at the current time step. In this situation, the Long Short Term Memory model will take in the previous 35 closing prices to predict the future closing price. This is done by processing the training data in the adequate format to feed into the neural network. 
Python contains the Google Tensorflow library which is compatible with the Keras library that holds the LSTM model, dropout, dense, and Sequential functions that were all used in creating the Long Short Term Memory model. The plot below displays the predicted closing price from the Long Short Term Memory model versus the real closing stock price. 
![image](https://user-images.githubusercontent.com/37299986/117738658-8b174400-b1ca-11eb-9099-2dab214847e2.png)
The Long Short Term Memory model obtained the following accuracy scores: 
- Mean absolute error = 2.72
- Mean squared error = 10.88
- Median absolute error = 2.49
- Explain variance score = 0.64
- R2 score = 0.59
