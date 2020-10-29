# Project Modeled off of TechWithTim Youtube Tutorial
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Reads csv file into pandas dataframe
postData = pd.read_csv("Daily_Demand_Forecasting_Orders.csv", sep=";")

# Replaces 'postData' with only the columns wanted
postData = postData[["Week of Month (1-5)", "Day (2-6)", "Non-urgent order", "Urgent order", "Order type A", "Order type B", "Order type C", "Fiscal sector orders", "Traffic controller sector orders", "Banking orders (1)", "Banking orders (2)", "Banking orders (3)", "Total orders"]]

# Sets variable for the data column being predicted
value_to_predict = "Total orders"

# Creates numpy array with all columns except the one being predicted
x = np.array(postData.drop([value_to_predict], 1))

# x array values were set in scientific notation, set here to float values
np.set_printoptions(suppress=True)

# x array values converted from float to int
x = x.astype(int)

# Creates numpy arrray with just the column containing post likes
y = np.array(postData[value_to_predict])

# Splits x and y datasets: x array training values, x array testing values, y array training values, and y array testing values
# Separates 10% of each dataset to be used for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

highest_R2 = 0

for i in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Initialize 'linear' variable to be linear regression model, then takes in training data
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    y_predict = linear.predict(x_test)
    R2 = r2_score(y_test, y_predict)
    if R2 > highest_R2:
        highest_R2 = R2
        with open("ordermodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_info = open("ordermodel.pickle", "rb")
linear = pickle.load(pickle_info)

R2 = r2_score(y_test, y_predict)
MSE = mean_squared_error(y_test, y_predict)

for i in range(len(y_predict) - 1):
    print(f"Predicted Orders: {y_predict[i]}    ", f"Actual Orders: {y_test[i]}")
