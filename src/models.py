import numpy as np


def ols_fit(S, y):
    """
    Closed-form solution for ordinary least squares.

    The design matrix S includes the intercept column.
    """
    return np.linalg.pinv(S.T @ S) @ S.T @ y


def ridge_fit(S, y, alpha):
    """
    Closed-form solution for ridge regression.

    The regularization term is applied only to the feature coefficients,
    while the intercept is left unpenalized.
    """
    n_features = S.shape[1]

    identity = np.eye(n_features)
    identity[0, 0] = 0

    return np.linalg.pinv(S.T @ S + alpha * identity) @ S.T @ y


def predict(S, w):
    """
    Linear prediction rule.
    """
    return S @ w


def mse(y_true, y_pred):
    """
    Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)