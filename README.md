# Stock Market Forecast Project

# Introduction
The purpose of this project is to investigate five different regression machine learning algorithms and see which one is best for predicting future closing stock prices. The five algorithms chosen vary from begginner implementation difficulty to advanced implementation difficulty. The five algorithms tested are as such: 
  - Moving Average (Beginner)
  - Linear Regression
  - K Nearest Neighbor
  - Auto ARIMA
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
