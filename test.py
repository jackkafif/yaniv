import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)  # Independent variable
y = 4 + 3 * x + np.random.randn(100, 1)  # Dependent variable with some noise

# Simple linear regression calculation
# Adding a bias (intercept) term
X_b = np.c_[np.ones((100, 1)), x]  # add x0 = 1 for each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Making predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add the intercept term
y_predict = X_new_b.dot(theta_best)

# Plotting the results
plt.plot(x, y, "b.")
plt.plot(X_new, y_predict, "r-")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.title("Simple Linear Regression")
plt.show()

# Print the coefficients
print("Estimated coefficients:")
print(f"Intercept: {theta_best[0][0]}")
print(f"Slope: {theta_best[1][0]}")
