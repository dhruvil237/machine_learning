import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
boston = datasets.load_boston()
#['data', 'target', 'feature_names', 'DESCR', 'filename']
boston_x = boston.data
boston_x_train = boston_x[:-70]
boston_x_test = boston_x[-30:]
boston_y_train = boston.target[:-70]
boston_y_test = boston.target[-30:]
model = linear_model.LinearRegression()
model.fit(boston_x_train,boston_y_train)
boston_y_predicted = model.predict(boston_x_test)
print("Mean squared error is:", mean_squared_error(boston_y_test, boston_y_predicted))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)
dic={}
for i in range(30):
    dic[boston_y_predicted[i]]=boston_y_test[i]
print(dic)
