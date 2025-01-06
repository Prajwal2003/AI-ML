import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 4, 2, 5, 6])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.title("Linear Regression Example")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.grid()
plt.show()
