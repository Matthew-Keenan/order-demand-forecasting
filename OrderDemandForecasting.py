import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plot 

# Reads csv file into pandas dataframe
orderData = pd.read_csv("Daily_Demand_Forecasting_Orders.csv", sep=";")

# Sets variable for the data column being predicted
value_to_predict = "Total orders"

# Creates numpy array with all columns except the one being predicted
x = np.array(orderData.drop([value_to_predict], 1))

# x array values were set in scientific notation, set here to float values
np.set_printoptions(suppress=True)

# x array values converted from float to int
x = x.astype(int)

# Creates numpy arrray with just the column containing post likes
y = np.array(orderData[value_to_predict])

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

# Set what data from dataset will be displayed on x and y axes
y_axis = orderData[value_to_predict]
x_axis = orderData["Order type B"]

# Plots data from given columns, labels the axes, and displays the scatterplot
plot.scatter(x_axis, y_axis)
plot.xlabel("Order Type B")
plot.ylabel("Total Orders")
plot.show()