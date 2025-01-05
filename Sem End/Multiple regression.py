import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
# Independent variables
X1 = np.array([1, 2, 3, 4, 5])  # Feature 1
X2 = np.array([5, 4, 3, 2, 1])  # Feature 2
X = np.column_stack((X1, X2))  # Combine into a 2D array

# Dependent variable
y = np.array([1.2, 2.8, 3.4, 4.5, 5.7])

# Create and fit the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Print model parameters
print("Coefficients:", model.coef_)  # Slopes for X1 and X2
print("Intercept:", model.intercept_)

# Plot the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y)), y, color="blue", label="Actual Values")  # Actual values
plt.plot(range(len(y_pred)), y_pred, color="red", marker="o", label="Predicted Values")  # Predicted values

# Add labels, title, and legend
plt.title("Multiple Linear Regression - Actual vs Predicted")
plt.xlabel("Data Point Index")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.grid()
plt.show()
