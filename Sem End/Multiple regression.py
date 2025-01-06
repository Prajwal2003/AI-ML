import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([5, 4, 3, 2, 1])
X = np.column_stack((X1, X2))

y = np.array([1.2, 2.8, 3.4, 4.5, 5.7])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y)), y, color="blue", label="Actual Values")
plt.plot(range(len(y_pred)), y_pred, color="red", marker="o", label="Predicted Values")
plt.title("Multiple Linear Regression - Actual vs Predicted")
plt.xlabel("Data Point Index")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.grid()
plt.show()
