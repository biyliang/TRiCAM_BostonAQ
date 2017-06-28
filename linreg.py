#For reading and manipulating tabular data
import numpy as np
import pandas as pd

#For plotting
import matplotlib.pyplot as plt
%matplotlib inline

#For basic statistical modeling
import scipy as sp
import sklearn as sk
import statsmodels.api as sm

df = pd.read_csv('ucidata.csv', sep=",", header=None)

#replaces question marks with "0"
for c in range(0, 127):
    for r in range(0, 1994):
        if df[c][r] == "?":
            df.set_value(r, c, 0)
            
#turns all columns that are numerics encoded as strings into floats
for c in range(4, 127):
    for r in range(0, 1994):
        if type(df[c][r]) == str:
            df.set_value(r, c, float(df[c][r]))

#takes out features columns
x = df.iloc[:, 4:126]

#takes of what we want to predict
target = df.iloc[:, 127]

#runs linear regression and returns summary of model

def reg_m(y, x):
    model = sm.OLS(y, x.astype(float)).fit()
    #fits simple ordinary least squares model
    predictions = model.predict(x)
    #makes predictions for y based on x
    return(model.summary())

x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.30, random_state=0)
print (x_train.shape, x_test.shape)


model_train = sm.OLS(y_train, x_train.astype(float)).fit()
#fits simple ordinary least squares model
y_pred_train = model.predict(x_train)

#fits simple ordinary least squares model
y_pred_test = model.predict(x_test)

#compute mean squared error
train_error = np.mean((y_train - y_pred_train)**2)
test_error = np.mean((y_test - y_pred_test)**2)

print ('error on training set:', train_error)
print ('error on testing set:', test_error)