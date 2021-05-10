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
The moving average model contained the following accuracy score: 
  - Mean absolute error = 3.32
  - Mean squared error = 17.03
  - Median absolute error = 2.89
  - Explain variance score = 0.47
  - R2 score = 0.36


