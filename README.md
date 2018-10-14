# Output1 and 2 Version 1

#Importing Library

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
#Reading Dataset
dataset = pd.read_csv('Book1.csv')

# Dividing Dataset in dependent and independent variables
x = dataset.iloc[:,[2,4,5]].values
y =dataset.iloc[:,9].values

# Divide dataset in train(75%) and test(25%)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.25)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train,y_train)

output1_pred = regressor.predict(x_test)
#rmse = np.sqrt(mean_squared_error(y_test, output1_pred))
#print("RMSE: %f" % (rmse))
