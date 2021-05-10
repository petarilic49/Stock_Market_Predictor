#PURPOSE: to investigate the different machine learning regression algorithms and see which is best for predicting stock market prices
import Moving_Average
import Linear_Regression
import KNearestNeighbor
import AutoARIMA
import LongShort
import pandas as pd 
from matplotlib import pyplot as plt

#We will use AMD stock prices for example
#Load the stock prices and other variables into the data variable by reading the .csv file
data = pd.read_csv('AMD.csv')
#Plot the closing price of the full dataset
plt.plot(data['Date'][:round(len(data)*0.7)], data['Close'][:round(len(data)*0.7)], label = 'Training Closing Price Data')
plt.plot(data['Date'][round(len(data)*0.7):len(data)], data['Close'][round(len(data)*0.7):len(data)], label = 'Testing Closing Price Data')
plt.title('Closing Price of Stock')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

#Call the individual algorithm functions to run and predict the future closing price for the testing data
Moving_Average.MovingAvg(data) 
# Linear_Regression.LinearReg(data)   
# KNearestNeighbor.Knearest(data)
# AutoARIMA.AutoARIMA(data)
# LongShort.LongShort(data)
