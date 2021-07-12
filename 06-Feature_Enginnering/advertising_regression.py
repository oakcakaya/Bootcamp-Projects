"""
There dependent and independent variables in Advertising.csv.
Sales is dependent variable whereas others are independent.
2 weight set are found by two different prediction method.
Answer the following questions about these weights.


b = 2.90, w1 = 0.04, w2 = 0.17, w3= 0.002

b = 1.70, w1 = 0.09, w2 = 0.20, w3= 0.017

1. Define the prediction function for the weights.
2. Calculate y_hat using the weights. (Predicted  y  values)
3. Calculate mse
4. Calculate rmse
5. Calculate mae

*Calculate the values above for both weight set. Campare and discuss the mse, rmse and mae values.
 Which method is better? The error of model is favorable?

"""
import numpy as np
import pandas as pd


def load_advertising():
    data = pd.read_csv("datasets/Advertising.csv")
    return data

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_advertising()
df.head()


##############################################
#1-Prediction Functions
##############################################
# b = 2.90, w1 = 0.04, w2 = 0.17, w3= 0.002
# y1_hat=2.90+0.04*x1+0.17*x2+0.002*x3
y1_hat = b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']




# b = 1.70, w1 = 0.09, w2 = 0.20, w3= 0.017
# y2_hat=1.70+0.09*x1+0.20*x2+0.017*x3
y2_hat = b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']


##############################################
#2- y_hat calculation
##############################################
# b = 2.90, w1 = 0.04, w2 = 0.17, w3= 0.002
# y=2.90+0.04*x1+0.17*x2+0.002*x3
y1_hat = b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']

b = 2.90
w1 = 0.04
w2 = 0.17
w3= 0.002

df["y1_hat"]=b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']
df.head()

# b = 1.70, w1 = 0.09, w2 = 0.20, w3= 0.017
# y=1.70+0.09*x1+0.20*x2+0.017*x3
y2_hat = b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']
b = 1.70
w1 = 0.09
w2 = 0.20
w3= 0.017
df["y2_hat"]=b + w1 * df['TV'] + w2 * df['radio'] + w3 * df['newspaper']
df.head()

##############################################
#2- mse calculation
##############################################
y1_mse = np.mean((df['sales'] - df['y1_hat'])**2)
y2_mse = np.mean((df['sales'] - df['y2_hat'])**2)
y1_mse
y2_mse

##############################################
#2- rmse calculation
##############################################
y1_rmse = np.sqrt(np.mean((df['sales'] - df['y1_hat'])**2))
y2_rmse = np.sqrt(np.mean((df['sales'] - df['y2_hat'])**2))

y1_rmse
y2_rmse

##############################################
#3-  mae calculation
##############################################

y1_mae = np.mean(np.abs(df['sales'] - df['y1_hat']))
y2_mae = np.mean(np.abs(df['sales'] - df['y2_hat']))

y1_mae
y2_mae


print("First set MSE:{}, RMSE:{}, MAE:{}\n"
      "Second set MSE:{}, RMSE:{}, MAE:{}\n".format(y1_mse,y1_rmse,y1_mae, y2_rmse, y2_rmse, y2_mae))

"""
For given prediction functions and weight sets, errors are lower in first one.
This function is more suitable for prediction since ii produces lower errors.
 
y1_hat=2.90+0.04*x1+0.17*x2+0.002*x3
y2_hat=1.70+0.09*x1+0.20*x2+0.017*x3

First set MSE:4.606008758799998, RMSE:2.1461614009202563, MAE:1.839058
Second set MSE:7.390060390030652, RMSE:7.390060390030652, MAE:6.217319999999997
"""
























