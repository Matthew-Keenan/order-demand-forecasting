# Project Modeled off of TechWithTim Youtube Tutorial
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

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

# Initialize 'linear' variable to be linear regression object, trains 'linear' with the two training sets
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# Forms predictions based on testing set
y_predict = linear.predict(x_test)

# Calculates the R2 Score and MSE given y_test values and y_predict values
R2 = r2_score(y_test, y_predict)
MSE = mean_squared_error(y_test, y_predict)

# Neatly displays data found
print(f"                R2 Value: {round(R2, 6)}")
print(f"Mean Squared Error (MSE): {round(MSE, 6)}")
print()
for i in range(len(y_predict) - 1):
    print(f"Predicted Orders: {round(y_predict[i], 2)} \t Actual Given Orders: {round(y_test[i], 2)}" )