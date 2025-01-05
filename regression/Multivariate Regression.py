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

# Generate predictions for plotting
X2_mean = np.mean(X2)  # Fix X2 as its mean value for 2D visualization
X_pred = np.column_stack((X1, np.full_like(X1, X2_mean)))  # Use X1 with X2 fixed
y_pred = model.predict(X_pred)

# Print model parameters
print("Coefficients:", model.coef_)  # Slopes for X1 and X2
print("Intercept:", model.intercept_)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X1, y, color="blue", label="Actual Values")  # Actual data points
plt.plot(X1, y_pred, color="red", label="Regression Line")  # Predicted values
plt.title("Multivariate Regression (2D Graph)")
plt.xlabel("Feature 1 (X1)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.grid()
plt.show()
