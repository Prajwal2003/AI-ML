import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
# X represents the independent variable, and y represents the dependent variable
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshape to 2D array for sklearn
y = np.array([3, 4, 2, 5, 6])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Print model parameters
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Plot the data and the regression line
plt.scatter(X, y, color="blue", label="Actual Data")  # Scatter plot for data points
plt.plot(X, y_pred, color="red", label="Regression Line")  # Line for predictions
plt.title("Linear Regression Example")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.grid()
plt.show()
