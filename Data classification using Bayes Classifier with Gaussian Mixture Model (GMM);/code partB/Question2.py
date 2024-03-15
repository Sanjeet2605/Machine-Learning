# importing importamt library
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading csv file
data = pd.read_csv("abalone.csv")
y = data["Rings"]
X = data.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# building model and fitting data
reg = LinearRegression()
reg.fit(X_train, y_train)


# predicting through built model
pred_1 = reg.predict(X_train)
pred_2 = reg.predict(X_test)

# function for calculating rmse


def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


# printing both of them
print("prediction accuracy on the training data using root mean squared error",
      round(rmse(pred_1, y_train), 3))
print("prediction accuracy on the test data using root mean squared error",
      round(rmse(pred_2, y_test), 3))

# scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data.
plt.scatter(y_test, pred_2, color="red", alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
