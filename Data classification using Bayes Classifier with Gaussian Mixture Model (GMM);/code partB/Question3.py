# importing important libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# reading csv file
data = pd.read_csv("abalone.csv")
# creating traing and testing data
y = data["Rings"]
X = data.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# finding correlation
print(data.corr())


# since shell weight has maximum correlation
x_train = X_train["Shell weight"]
x_test = X_test["Shell weight"]

# reshaing for modelling data
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

x_test = np.array(x_test)
x_test = x_test.reshape(-1, 1)
y_test = np.array(y_test)
y_test = y_test.reshape(-1, 1)

# function for findign rmse


def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


# for every p in (2,3,4,5) making polyreg model and predicting training data
P = [2, 3, 4, 5]
RMSE1 = []
for p in P:
    poly_features = PolynomialFeatures(p)
    x_poly = poly_features.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    predict1 = regressor.predict(poly_features.fit_transform(x_train))
    print("the prediction accuracy on the training data for the different values of degree ofthe polynomial p =",
          p, " using root mean squared error (RMSE) is ", rmse(predict1, y_train))
    RMSE1.append(rmse(predict1, y_train))
plt.bar(P, RMSE1)
plt.xlabel("value of p")
plt.ylabel("RMSE value")
plt.show()

# since best prediction is at p=5
p = 5
poly_features = PolynomialFeatures(p)
x_poly = poly_features.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)
predict1 = regressor.predict(poly_features.fit_transform(x_test))
plt.scatter(x_test, predict1)
plt.xlabel("attribute")
plt.ylabel("predicted")
plt.show()

# for every p in (2,3,4,5) making polyreg model and predicting test data
P = [2, 3, 4, 5]
RMSE2 = []
for p in P:
    poly_features = PolynomialFeatures(p)
    x_poly = poly_features.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    predict1 = regressor.predict(poly_features.fit_transform(x_test))
    print("the prediction accuracy on the test data for the different values of degree ofthe polynomial p =",
          p, " using root mean squared error (RMSE) is ", rmse(predict1, y_test))
    RMSE2.append(rmse(predict1, y_test))
plt.bar(P, RMSE2)
plt.xlabel("value of p")
plt.ylabel("RMSE value")
plt.show()

# since best prediction is at p=5
p = 5
poly_features = PolynomialFeatures(p)
x_poly = poly_features.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)
predict1 = regressor.predict(poly_features.fit_transform(x_test))
plt.scatter(y_test, predict1)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
