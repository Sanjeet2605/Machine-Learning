# importing important libraries
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

train = pd.concat([X_train, y_train], axis=1, join='inner')
test = pd.concat([X_test, y_test], axis=1, join='inner')

# saving the training and testing datasets as csv files
train.to_csv('abalone-train.csv', index=False)
test.to_csv('abalone-test.csv', index=False)

# finding correlation
print(data.corr())


# since shell weight ihas maximum correlation
x_train = X_train["Shell weight"]
x_test = X_test["Shell weight"]

# model building
reg = LinearRegression()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

x_test = np.array(x_test)
x_test = x_test.reshape(-1, 1)
y_test = np.array(y_test)
y_test = y_test.reshape(-1, 1)
reg.fit(x_train, y_train)

# predicting x_test and using it plot line
y = reg.predict(x_test)
# scattter plot b/w selected attribute and target attribute
plt.scatter(x_train, y_train)
plt.plot(x_test, y, color="r")
plt.xlabel("Shell weight")
plt.ylabel("Rings")
plt.show()

# predicting using obtained model
pred_1 = reg.predict(x_train)
pred_2 = reg.predict(x_test)

# calculating rmse value


def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


print("prediction accuracy on the training data using root mean squared error",
      round(rmse(pred_1, y_train), 3))
print("prediction accuracy on the test data using root mean squared error",
      round(rmse(pred_2, y_test), 3))

# scatter plot of actual Rings (x-axis) vs predicted Rings (y-axis) on the test data.
plt.scatter(y_test, pred_2, color="red", alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.show()
