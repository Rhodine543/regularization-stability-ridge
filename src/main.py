import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import ols_fit, ridge_fit, predict, mse
from stability import compute_stability


# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv("data/boston.csv")

y = data["MEDV"].values
X = data.drop(columns=["MEDV"]).values

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# -----------------------------
# Train-test split
# -----------------------------

np.random.seed(42)

indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -----------------------------
# Standardization
# -----------------------------

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("\nMean after standardization:")
print(X_train.mean(axis=0))

print("\nStd after standardization:")
print(X_train.std(axis=0))


# -----------------------------
# Add intercept column
# -----------------------------

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

print("\nNew shape with intercept:", X_train.shape)


# -----------------------------
# Single comparison: OLS vs Ridge
# -----------------------------

w_ols = ols_fit(X_train, y_train)

y_train_pred_ols = predict(X_train, w_ols)
y_test_pred_ols = predict(X_test, w_ols)

train_mse_ols = mse(y_train, y_train_pred_ols)
test_mse_ols = mse(y_test, y_test_pred_ols)

print("\nOrdinary Least Squares")
print("Train MSE:", train_mse_ols)
print("Test MSE:", test_mse_ols)


alpha = 1.0

w_ridge = ridge_fit(X_train, y_train, alpha)

y_train_pred_ridge = predict(X_train, w_ridge)
y_test_pred_ridge = predict(X_test, w_ridge)

train_mse_ridge = mse(y_train, y_train_pred_ridge)
test_mse_ridge = mse(y_test, y_test_pred_ridge)

print(f"\nRidge Regression alpha={alpha}")
print("Train MSE:", train_mse_ridge)
print("Test MSE:", test_mse_ridge)


# -----------------------------
# Alpha loop with stability
# -----------------------------

alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

results = []

for alpha in alphas:

    if alpha == 0:
        w = ols_fit(X_train, y_train)
    else:
        w = ridge_fit(X_train, y_train, alpha)

    y_train_pred = predict(X_train, w)
    y_test_pred = predict(X_test, w)

    train_error = mse(y_train, y_train_pred)
    test_error = mse(y_test, y_test_pred)

    stability = compute_stability(X_train, y_train, X_test, alpha)

    results.append({
        "alpha": alpha,
        "train_mse": train_error,
        "test_mse": test_error,
        "stability": stability
    })


results_df = pd.DataFrame(results)

print("\nResults across alpha values:")
print(results_df)


# -----------------------------
# Save results
# -----------------------------

os.makedirs("results/plots", exist_ok=True)

results_df.to_csv("results/results_with_stability.csv", index=False)


# -----------------------------
# Plot train/test error
# -----------------------------

plot_alphas = results_df["alpha"].replace(0, 1e-4)

plt.figure()
plt.plot(plot_alphas, results_df["train_mse"], marker="o", label="Train MSE")
plt.plot(plot_alphas, results_df["test_mse"], marker="o", label="Test MSE")
plt.xscale("log")
plt.xlabel("Regularization strength alpha")
plt.ylabel("Mean squared error")
plt.title("Train and Test Error vs Regularization")
plt.legend()
plt.savefig("results/plots/error_vs_alpha.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Plot stability
# -----------------------------

plt.figure()
plt.plot(plot_alphas, results_df["stability"], marker="o")
plt.xscale("log")
plt.xlabel("Regularization strength alpha")
plt.ylabel("Average absolute prediction change")
plt.title("Stability vs Regularization")
plt.savefig("results/plots/stability_vs_alpha.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Dataset size extension
# -----------------------------

sizes = [50, 100, 200, len(X_train)]

alpha_fixed = 1.0

size_results = []

for size in sizes:

    X_sub = X_train[:size]
    y_sub = y_train[:size]

    w = ridge_fit(X_sub, y_sub, alpha_fixed)

    y_train_pred = predict(X_sub, w)
    y_test_pred = predict(X_test, w)

    train_error = mse(y_sub, y_train_pred)
    test_error = mse(y_test, y_test_pred)

    stability = compute_stability(X_sub, y_sub, X_test, alpha_fixed)

    size_results.append({
        "training_size": size,
        "alpha": alpha_fixed,
        "train_mse": train_error,
        "test_mse": test_error,
        "stability": stability
    })


size_df = pd.DataFrame(size_results)

print("\nDataset size experiment:")
print(size_df)

size_df.to_csv("results/dataset_size_results.csv", index=False)


# -----------------------------
# Plot stability vs dataset size
# -----------------------------

plt.figure()
plt.plot(size_df["training_size"], size_df["stability"], marker="o")
plt.xlabel("Training set size")
plt.ylabel("Average absolute prediction change")
plt.title("Stability vs Training Set Size")
plt.savefig("results/plots/stability_vs_dataset_size.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Plot test error vs dataset size
# -----------------------------

plt.figure()
plt.plot(size_df["training_size"], size_df["test_mse"], marker="o")
plt.xlabel("Training set size")
plt.ylabel("Test MSE")
plt.title("Test Error vs Training Set Size")
plt.savefig("results/plots/test_error_vs_dataset_size.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nResults and plots saved successfully.")