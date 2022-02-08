import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Imports from SKlearn Linear Regression Model
from sklearn import linear_model

# This saves csv honey bee date provided from code academy 
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")


#prints cvc data head
print(df.head)

#saves prod per year as a mean value from the index
prod_per_year = df.groupby('year').totalprod.mean().reset_index()


#X is years overtime
X = df["year"]
#We reshape the paramater 
X = X.values.reshape(-1,1)

#Y is totalproduction of honey
y = df["totalprod"]


#Creates Honey Best line of fit from current data, using gradient descent with the model from Linear Regression Pckg
honeyLinearModel = linear_model.LinearRegression()

honeyLinearModel.fit(X,y)

#Prints the slope of the line of best fit
print(honeyLinearModel.coef_[0])

#Predict the values of total production with the data from X
y_predict = honeyLinearModel.predict(X)

#Plots scatter and best line of fit
plt.plot(X, y_predict)

plt.scatter(X, y)
plt.show()

#Predicts future honey production with current data
X_future = np.array(range(2013,2050))
X_future = X_future.reshape(-1,1)

future_predict = honeyLinearModel.predict(X_future)

plt.plot(X_future, future_predict)
plt.show()

