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

# function to calculate rmse


def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


# for every p in (2,3,4,5) making polyreg model and predicting training data
P = [2, 3, 4, 5]
RMSE1 = []
for p in P:
    poly_features = PolynomialFeatures(p)
    x_poly = poly_features.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    predict1 = regressor.predict(poly_features.fit_transform(X_train))
    print("the prediction accuracy on the training data for the different values of degree ofthe polynomial p =",
          p, " using root mean squared error (RMSE) is ", rmse(predict1, y_train))
    RMSE1.append(rmse(predict1, y_train))
plt.bar(P, RMSE1)
plt.xlabel("value of p")
plt.ylabel("RMSE value")
plt.show()


# for every p in (2,3,4,5) making polyreg model and predicting test data
P = [2, 3, 4, 5]
RMSE1 = []
for p in P:
    poly_features = PolynomialFeatures(p)
    x_poly = poly_features.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    predict1 = regressor.predict(poly_features.fit_transform(X_test))
    print("the prediction accuracy on the training data for the different values of degree ofthe polynomial p =",
          p, " using root mean squared error (RMSE) is ", rmse(predict1, y_test))
    RMSE1.append(rmse(predict1, y_test))
plt.bar(P, RMSE1)
plt.xlabel("value of p")
plt.ylabel("RMSE value")
plt.show()


# since p=2 has lowest rsme
p = 2
poly_features = PolynomialFeatures(p)
x_poly = poly_features.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)
predict1 = regressor.predict(poly_features.fit_transform(X_test))
plt.scatter(y_test, predict1)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
