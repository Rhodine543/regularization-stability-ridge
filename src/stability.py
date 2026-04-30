import numpy as np
from models import ols_fit, ridge_fit, predict


def fit_model(S, y, alpha):
    """
    Fits either ordinary least squares or ridge regression.

    alpha = 0 means unregularized least squares.
    alpha > 0 means ridge regression.
    """
    if alpha == 0:
        return ols_fit(S, y)
    else:
        return ridge_fit(S, y, alpha)


def compute_stability(S_train, y_train, S_test, alpha):
    """
    Estimates algorithmic stability using leave-one-out removal.

    For each training point:
    1. Remove the point.
    2. Retrain the model.
    3. Measure the average absolute change in predictions on a fixed test set.

    
    """

    # Train baseline model on the full training set
    w_full = fit_model(S_train, y_train, alpha)
    base_pred = predict(S_test, w_full)

    changes = []

    for i in range(S_train.shape[0]):

        # Remove the i-th training observation
        S_minus_i = np.delete(S_train, i, axis=0)
        y_minus_i = np.delete(y_train, i)

        # Retrain model without the i-th observation
        w_minus_i = fit_model(S_minus_i, y_minus_i, alpha)

        # Predict on the same fixed test set
        new_pred = predict(S_test, w_minus_i)

        # L1 stability metric: average absolute prediction change
        diff = np.mean(np.abs(new_pred - base_pred))

        changes.append(diff)

    return np.mean(changes)