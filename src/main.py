import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import ols_fit, ridge_fit, predict, mse
from stability import compute_stability


# ============================================================
# 1. Load and inspect the dataset
# ============================================================

data = pd.read_csv(
    "data/communities.data",
    header=None,
    na_values="?"
)

print("Original data shape:", data.shape)


# ============================================================
# 2. Data preprocessing
# ============================================================

# The first columns contain identifiers such as state, county,
# community code, community name, and fold. These are not used
# as predictive variables in the regression model.
data = data.drop(columns=[0, 1, 2, 3, 4])

print("Shape after dropping ID columns:", data.shape)


# Check missing values before cleaning
missing = data.isna().sum()

print("\nColumns with most missing values:")
print(missing.sort_values(ascending=False).head(10))

print("\nTotal missing values before cleaning:", missing.sum())


# Columns with more than 50% missing values are removed.
# These variables contain too little information to be useful.
missing_fraction = data.isna().mean()
threshold = 0.5

cols_to_drop = missing_fraction[missing_fraction > threshold].index

print("\nNumber of columns dropped:", len(cols_to_drop))

data = data.drop(columns=cols_to_drop)

print("Shape after dropping high-missing columns:", data.shape)


# Remaining missing values are replaced using column means.
data = data.fillna(data.mean())

print("Remaining missing values after cleaning:", data.isna().sum().sum())
print("Final cleaned dataset shape:", data.shape)


# ============================================================
# 3. Define features and target
# ============================================================

# The last column is the target variable: ViolentCrimesPerPop.
# The remaining columns are used as predictors.
y = data.iloc[:, -1].values
X = data.iloc[:, :-1].values

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

print("\nTarget summary:")
print(pd.Series(y).describe())


# ============================================================
# 4. Train-test split
# ============================================================

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


# ============================================================
# 5. Standardization
# ============================================================

# The mean and standard deviation are computed only on the
# training set to avoid data leakage.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# Add intercept column
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

print("Shape after adding intercept:", X_train.shape)


# ============================================================
# 6. Prepare output folders
# ============================================================

# A fresh plots folder is created at each run so that old plots
# do not remain mixed with the current results.
if os.path.exists("results/plots_communities"):
    shutil.rmtree("results/plots_communities")

os.makedirs("results/plots_communities", exist_ok=True)


# ============================================================
# 7. Training and test error for different regularization levels
# ============================================================

alphas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

error_results = []

for alpha in alphas:

    if alpha == 0:
        w = ols_fit(X_train, y_train)
        model_name = "OLS"
    else:
        w = ridge_fit(X_train, y_train, alpha)
        model_name = "Ridge"

    y_train_pred = predict(X_train, w)
    y_test_pred = predict(X_test, w)

    train_error = mse(y_train, y_train_pred)
    test_error = mse(y_test, y_test_pred)

    error_results.append({
        "alpha": alpha,
        "model": model_name,
        "train_mse": train_error,
        "test_mse": test_error
    })


error_df = pd.DataFrame(error_results)

print("\nTraining and test error results:")
print(error_df)

error_df.to_csv(
    "results/communities_error_results.csv",
    index=False
)


# Plot train and test error against alpha
plot_alphas = error_df["alpha"].replace(0, 1e-4)

plt.figure(figsize=(8, 5))
plt.plot(plot_alphas, error_df["train_mse"], marker="o", label="Train MSE")
plt.plot(plot_alphas, error_df["test_mse"], marker="o", label="Test MSE")
plt.xscale("log")
plt.xlabel("Regularization strength alpha")
plt.ylabel("Mean squared error")
plt.title("Train and Test Error vs Regularization")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/plots_communities/error_vs_alpha.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Separate test error plot, since it is one of the required outputs.
plt.figure(figsize=(8, 5))
plt.plot(plot_alphas, error_df["test_mse"], marker="o")
plt.xscale("log")
plt.xlabel("Regularization strength alpha")
plt.ylabel("Test MSE")
plt.title("Test Error vs Regularization")
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/plots_communities/test_error_vs_alpha.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. Stability estimation
# ============================================================

# Stability is estimated by removing one training observation,
# retraining the model, and measuring the average absolute
# change in predictions on the fixed test set.
stability_results = []

for alpha in alphas:

    stability = compute_stability(
        X_train,
        y_train,
        X_test,
        alpha
    )

    stability_results.append({
        "alpha": alpha,
        "stability": stability
    })


stability_df = pd.DataFrame(stability_results)

print("\nStability results:")
print(stability_df)

stability_df.to_csv(
    "results/communities_stability_results.csv",
    index=False
)


# Combine prediction error and stability results
results_df = pd.merge(error_df, stability_df, on="alpha")

print("\nCombined results:")
print(results_df)

results_df.to_csv(
    "results/communities_results_with_stability.csv",
    index=False
)


# Plot stability against alpha
plot_alphas = results_df["alpha"].replace(0, 1e-4)

plt.figure(figsize=(8, 5))
plt.plot(plot_alphas, results_df["stability"], marker="o")
plt.xscale("log")
plt.xlabel("Regularization strength alpha")
plt.ylabel("Average absolute prediction change")
plt.title("Stability vs Regularization")
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/plots_communities/stability_vs_alpha.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. Relationship between stability and generalization
# ============================================================

# Since stability is measured as prediction change, lower values
# correspond to more stable models. The correlation with test MSE
# is used as a numerical summary of the stability-generalization link.
correlation = results_df["stability"].corr(results_df["test_mse"])

print("\nCorrelation between stability and test error:", correlation)

pd.DataFrame({
    "correlation_stability_test_mse": [correlation]
}).to_csv(
    "results/communities_stability_test_error_correlation.csv",
    index=False
)


# Plot stability directly against test error
plot_df = results_df.sort_values(by="stability")

plt.figure(figsize=(8, 5))
plt.plot(plot_df["stability"], plot_df["test_mse"], marker="o")

for _, row in plot_df.iterrows():
    plt.annotate(
        f"alpha={row['alpha']}",
        (row["stability"], row["test_mse"]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=8
    )

plt.xlabel("Stability: average prediction change")
plt.ylabel("Test MSE")
plt.title("Stability vs Test Error")
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/plots_communities/stability_vs_test_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nMain experiments completed successfully.")


# ============================================================
# 10. Extension: Effect of dataset size
# ============================================================

sizes = [100, 300, 600, 1000, len(X_train)]
alpha_fixed = 1.0

size_results = []

for size in sizes:

    # Use a subset of the training data
    X_sub = X_train[:size]
    y_sub = y_train[:size]

    # Train ridge model
    w = ridge_fit(X_sub, y_sub, alpha_fixed)

    # Predictions
    y_train_pred = predict(X_sub, w)
    y_test_pred = predict(X_test, w)

    # Errors
    train_error = mse(y_sub, y_train_pred)
    test_error = mse(y_test, y_test_pred)

    # Stability
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

size_df.to_csv(
    "results/communities_dataset_size_results.csv",
    index=False
)

#Plot Stability against dataset size 
plt.figure(figsize=(8, 5))

plt.plot(size_df["training_size"], size_df["stability"], marker="o")

plt.xlabel("Training set size")
plt.ylabel("Average absolute prediction change")
plt.title("Stability vs Training Set Size")
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/plots_communities/stability_vs_dataset_size.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


#Test error against dataset size
plt.figure(figsize=(8, 5))

plt.plot(size_df["training_size"], size_df["test_mse"], marker="o")

plt.xlabel("Training set size")
plt.ylabel("Test MSE")
plt.title("Test Error vs Training Set Size")
plt.grid(True, alpha=0.3)

plt.savefig(
    "results/plots_communities/test_error_vs_dataset_size.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()