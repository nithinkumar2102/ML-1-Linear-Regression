import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ==================================================
# STEP 1: LOAD DATASET
# ==================================================

print("Loading California Housing Dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset Shape:", df.shape)


# ==================================================
# STEP 2: SEPARATE FEATURES AND TARGET
# ==================================================

X = df.drop("MedHouseVal", axis=1).to_numpy()
y = df["MedHouseVal"].to_numpy().reshape(-1, 1)

print("Features Shape:", X.shape)


# ==================================================
# STEP 3: TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==================================================
# STEP 4: FEATURE SCALING
# ==================================================

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("Feature Scaling Completed!")


# ==================================================
# STEP 5: LINEAR REGRESSION FUNCTIONS
# ==================================================

# Add bias column
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


# Hypothesis function
def hypothesis(X, theta):
    return X @ theta


# Mean Squared Error cost function
def compute_cost(X, y, theta):

    predictions = hypothesis(X, theta)

    errors = predictions - y

    cost = np.mean(errors ** 2)

    return cost


# Gradient Descent algorithm
def gradient_descent(X, y, learning_rate, iterations):

    m, n = X.shape

    theta = np.zeros((n, 1))

    cost_history = []

    for i in range(iterations):

        predictions = hypothesis(X, theta)

        errors = predictions - y

        gradient = (2 / m) * (X.T @ errors)

        theta = theta - learning_rate * gradient

        cost = compute_cost(X, y, theta)

        cost_history.append(cost)

    return theta, cost_history


print("\nLINEAR REGRESSION FUNCTIONS CREATED SUCCESSFULLY!")


# ==================================================
# STEP 6: MULTIPLE-FEATURE MODEL TRAINING
# ==================================================

X_train_bias = add_bias(X_train)
X_test_bias = add_bias(X_test)

print("\nTraining Multiple-Feature Linear Regression...")

theta, cost_history = gradient_descent(
    X_train_bias,
    y_train,
    learning_rate=0.05,
    iterations=3000
)

print("\nMODEL TRAINING COMPLETED!")

print("\nLearned Parameters:")
print(theta)

print("\nInitial Cost:", cost_history[0])
print("Final Cost:", cost_history[-1])


# ==================================================
# STEP 7: MAKE PREDICTIONS
# ==================================================

y_pred = hypothesis(X_test_bias, theta)

print("\nPREDICTIONS COMPLETED!")

print("\nFirst 10 Actual Values:")
print(y_test[:10].flatten())

print("\nFirst 10 Predicted Values:")
print(y_pred[:10].flatten())


# ==================================================
# STEP 8: MANUAL MSE CALCULATION
# ==================================================

mse = np.mean((y_test - y_pred) ** 2)

print("\nMANUAL TEST MSE:")
print(mse)


# ==================================================
# STEP 9: MANUAL R2 SCORE CALCULATION
# ==================================================

ss_total = np.sum(
    (y_test - np.mean(y_test)) ** 2
)

ss_residual = np.sum(
    (y_test - y_pred) ** 2
)

r2_score = 1 - (ss_residual / ss_total)

print("\nMANUAL R2 SCORE:")
print(r2_score)


# ==================================================
# STEP 10: SCIKIT-LEARN COMPARISON
# ==================================================

sk_model = LinearRegression()

sk_model.fit(X_train, y_train)

sk_predictions = sk_model.predict(X_test)

sk_mse = np.mean(
    (y_test - sk_predictions) ** 2
)

sk_ss_total = np.sum(
    (y_test - np.mean(y_test)) ** 2
)

sk_ss_residual = np.sum(
    (y_test - sk_predictions) ** 2
)

sk_r2 = 1 - (sk_ss_residual / sk_ss_total)

print("\nSCIKIT-LEARN RESULTS")
print("-------------------------")

print("Scikit-learn MSE:", sk_mse)
print("Scikit-learn R2:", sk_r2)

print("\nMODEL COMPARISON")
print("-------------------------")

print("Scratch Model MSE:", mse)
print("Scikit-learn MSE:", sk_mse)

print("Scratch Model R2:", r2_score)
print("Scikit-learn R2:", sk_r2)


# ==================================================
# STEP 11: COST REDUCTION GRAPH
# ==================================================

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

plt.title("Gradient Descent Cost Reduction")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/cost_convergence.png")

print("\nCOST GRAPH SAVED SUCCESSFULLY!")

plt.show()

plt.close()


# ==================================================
# STEP 12: SINGLE-FEATURE LINEAR REGRESSION
# ==================================================

print("\nSINGLE-FEATURE LINEAR REGRESSION")

# Select MedInc feature only
X_train_single = X_train[:, [0]]

X_test_single = X_test[:, [0]]

# Add bias column
X_train_single_bias = add_bias(X_train_single)

X_test_single_bias = add_bias(X_test_single)

# Train single-feature model
single_theta, single_cost_history = gradient_descent(
    X_train_single_bias,
    y_train,
    learning_rate=0.05,
    iterations=3000
)

# Make predictions
single_predictions = hypothesis(
    X_test_single_bias,
    single_theta
)

# Calculate MSE
single_mse = np.mean(
    (y_test - single_predictions) ** 2
)

print("\nSingle-Feature Parameters:")
print(single_theta)

print("\nSingle-Feature Test MSE:")
print(single_mse)

print("\nSINGLE-FEATURE REGRESSION COMPLETED!")


# ==================================================
# FINAL STATUS
# ==================================================

print("\n====================================")
print("ALL ASSIGNMENT STEPS COMPLETED!")
print("====================================")