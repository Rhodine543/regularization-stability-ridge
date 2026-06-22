import numpy as np


def generate_synthetic_data(
    n_samples=200,
    n_features=80,
    noise_std=1.0,
    random_state=42
):

    np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)

    # Strong multicollinearity
    for i in range(40, 80):
        X[:, i] = X[:, i - 40] + 0.01 * np.random.randn(n_samples)

    true_w = np.random.randn(n_features)

    noise = noise_std * np.random.randn(n_samples)

    y = X @ true_w + noise

    return X, y, true_w