import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import ols_fit, ridge_fit, predict, mse
from stability import compute_stability
from synthetic_data import generate_synthetic_data


ALPHAS = [0, 0.001, 0.01, 0.1, 1, 10, 50, 100]


# ============================================================
# Utility functions
# ============================================================

def prepare_output_folder(dataset_name):
    output_dir = f"results/{dataset_name}/plots"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def train_test_split_from_scratch(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    indices = np.random.permutation(len(X))
    split = int((1 - test_size) * len(X))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_from_train(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def add_intercept(X_train, X_test):
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    return X_train, X_test


# ============================================================
# Main experiment function
# ============================================================

def run_regularization_stability_experiment(X, y, dataset_name):
    print(f"\n================ {dataset_name.upper()} DATASET ================")

    plots_dir = prepare_output_folder(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split_from_scratch(X, y)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    X_train, X_test = standardize_from_train(X_train, X_test)
    X_train, X_test = add_intercept(X_train, X_test)

    results = []

    for alpha in ALPHAS:
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

        stability = compute_stability(X_train, y_train, X_test, alpha)

        results.append({
            "alpha": alpha,
            "model": model_name,
            "train_mse": train_error,
            "test_mse": test_error,
            "stability": stability
        })

    results_df = pd.DataFrame(results)

    print("\nResults:")
    print(results_df)

    os.makedirs(f"results/{dataset_name}", exist_ok=True)

    results_df.to_csv(
        f"results/{dataset_name}/results_with_stability.csv",
        index=False
    )

    correlation = results_df["stability"].corr(results_df["test_mse"])

    print("\nCorrelation between stability and test error:", correlation)

    pd.DataFrame({
        "correlation_stability_test_mse": [correlation]
    }).to_csv(
        f"results/{dataset_name}/stability_test_error_correlation.csv",
        index=False
    )

    plot_alphas = results_df["alpha"].replace(0, 1e-4)

    # --------------------------------------------------------
    # Plot 1: Train and test error vs regularization
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(plot_alphas, results_df["train_mse"], marker="o", label="Train MSE")
    plt.plot(plot_alphas, results_df["test_mse"], marker="o", label="Test MSE")
    plt.xscale("log")
    plt.xlabel("Regularization strength alpha")
    plt.ylabel("Mean squared error")
    plt.title(f"{dataset_name}: Train and Test Error vs Regularization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/error_vs_alpha.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Plot 2: Stability vs regularization
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(plot_alphas, results_df["stability"], marker="o")
    plt.xscale("log")
    plt.xlabel("Regularization strength alpha")
    plt.ylabel("Average absolute prediction change")
    plt.title(f"{dataset_name}: Stability vs Regularization")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/stability_vs_alpha.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Plot 3: Stability vs test error
    # --------------------------------------------------------
    plt.figure(figsize=(12, 7))

    important_alphas = [0, 10, 50, 100]

    point_sizes = [
        380 if alpha in important_alphas else 130
        for alpha in results_df["alpha"]
    ]

    plt.scatter(
        results_df["stability"],
        results_df["test_mse"],
        s=point_sizes
    )

    label_offsets = {
    0: (18, 8),
    10: (8, 14),
    50: (15, 12),
    100: (15, 12)
}

    for _, row in results_df.iterrows():
        alpha = row["alpha"]

        if alpha not in important_alphas:
            continue

        dx, dy = label_offsets.get(alpha, (12, 10))

        label = f"α={int(alpha)}" if alpha >= 1 else f"α={alpha}"

        plt.annotate(
            label,
            (row["stability"], row["test_mse"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=12,
            fontweight="bold"
        )

    

    plt.xlabel("Average prediction change (lower = more stable)", fontsize=13)
    plt.ylabel("Test MSE", fontsize=13)
    plt.title(f"{dataset_name}: Stability vs Test Error", fontsize=18)
    plt.grid(True, alpha=0.3)

    plt.savefig(
        f"{plots_dir}/stability_vs_test_error.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return X_train, X_test, y_train, y_test, results_df


# ============================================================
# Load Communities and Crime dataset
# ============================================================

def load_communities_data():
    data = pd.read_csv(
        "data/communities.data",
        header=None,
        na_values="?"
    )

    print("Original Communities shape:", data.shape)

    data = data.drop(columns=[0, 1, 2, 3, 4])

    missing_fraction = data.isna().mean()
    cols_to_drop = missing_fraction[missing_fraction > 0.5].index
    data = data.drop(columns=cols_to_drop)

    data = data.fillna(data.mean())

    print("Cleaned Communities shape:", data.shape)

    y = data.iloc[:, -1].values
    X = data.iloc[:, :-1].values

    print("Target summary:")
    print(pd.Series(y).describe())

    return X, y


# ============================================================
# Dataset size extension
# ============================================================

def run_dataset_size_extension(
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_name="communities"
):
    print("\n================ DATASET SIZE EXTENSION ================")

    sizes = [100, 300, 600, 1000, len(X_train)]
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

    print(size_df)

    size_df.to_csv(
        f"results/{dataset_name}/dataset_size_results.csv",
        index=False
    )

    plots_dir = f"results/{dataset_name}/plots"

    # Stability vs dataset size
    plt.figure(figsize=(8, 5))
    plt.plot(size_df["training_size"], size_df["stability"], marker="o")
    plt.xlabel("Training set size")
    plt.ylabel("Average absolute prediction change")
    plt.title("Stability vs Training Set Size")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        f"{plots_dir}/stability_vs_dataset_size.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Test error vs dataset size
    plt.figure(figsize=(8, 5))
    plt.plot(size_df["training_size"], size_df["test_mse"], marker="o")
    plt.xlabel("Training set size")
    plt.ylabel("Test MSE")
    plt.title("Test Error vs Training Set Size")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        f"{plots_dir}/test_error_vs_dataset_size.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# Run full project
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Synthetic dataset: controlled experiment
    # --------------------------------------------------------
    X_syn, y_syn, true_w = generate_synthetic_data()

    print("\nSynthetic X shape:", X_syn.shape)
    print("Synthetic y shape:", y_syn.shape)

    run_regularization_stability_experiment(
        X_syn,
        y_syn,
        dataset_name="synthetic"
    )

    # --------------------------------------------------------
    # 2. Communities and Crime dataset: real-world validation
    # --------------------------------------------------------
    X_comm, y_comm = load_communities_data()

    X_train_comm, X_test_comm, y_train_comm, y_test_comm, communities_results = (
        run_regularization_stability_experiment(
            X_comm,
            y_comm,
            dataset_name="communities"
        )
    )

    # --------------------------------------------------------
    # 3. Extension: effect of dataset size
    # --------------------------------------------------------
    run_dataset_size_extension(
        X_train_comm,
        X_test_comm,
        y_train_comm,
        y_test_comm,
        dataset_name="communities"
    )

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()

    