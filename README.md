# Regularization, Stability and Generalization

This project investigates the relationship between **regularization strength**, **algorithmic stability**, and **generalization performance**, following the ideas introduced in:

> Bousquet, O., & Elisseeff, A. (2002). *Stability and Generalization*. Journal of Machine Learning Research.

The project  studies how **L2 regularization (Ridge Regression)** influences the stability of a learning algorithm and its ability to generalize to unseen data.

---

## Objective

The goal of this project is to investigate the relationship between:

- Regularization strength (α)
- Algorithmic stability
- Generalization performance

The study folllows the implemenation of two different datsets;

- a **synthetic regression dataset**, where the data-generating process is controlled,
- a **real-world regression dataset** (Communities and Crime), to validate the observations in practice.

---

## Methodology

For each value of the regularization parameter α:

1. Split the dataset into training and testing sets.
2. Standardize the predictors using statistics computed from the training set.
3. Add an intercept term.
4. Train:
   - Ordinary Least Squares (α = 0)
   - Ridge Regression (α > 0)
5. Compute:
   - Training Mean Squared Error
   - Test Mean Squared Error
6. Estimate algorithmic stability by:
   - removing one training observation,
   - retraining the model,
   - measuring the average change in predictions on a fixed test set.

---

## Datasets

### 1. Synthetic Dataset

The synthetic dataset is generated from a known linear model with Gaussian noise.

Characteristics:

Dataset characteristics:

- 200 observations
- 80 predictor variables
- 40 independent Gaussian features
- 40 highly correlated features created as noisy copies of the first 40 variables
- Linear target generated with additive Gaussian noise
The deliberate introduction of multicollinearity makes the ordinary least squares solution unstable, providing an ideal setting for evaluating the stabilizing effect of ridge regression.

---

### 2. Communities and Crime Dataset

The Communities and Crime dataset is obtained from the UCI Machine Learning Repository.

Preprocessing steps:

- Removal of identifier variables
- Removal of variables with excessive missing values
- Mean imputation for remaining missing values
- Feature standardization

This dataset serves as a real-world validation of the observations obtained from the synthetic experiment.

---

## Stability Metric

Algorithmic stability is estimated using a leave-one-out procedure.

For each training observation:

- remove one observation,
- retrain the model,
- predict on the same test set,
- compute the average absolute prediction difference.

Smaller prediction changes indicate a more stable learning algorithm.

---

## Results

For each dataset the following analyses are performed:

- Train Error vs Regularization
- Test Error vs Regularization
- Stability vs Regularization
- Stability vs Test Error

The project compares how regularization affects both prediction accuracy and algorithmic stability under controlled and real-world conditions.

---

## Repository Structure

```
regularization_stability_project/

│

├── data/
│   └── communities.data
│
├── results/
│   ├── synthetic/
│   └── communities/
│
├── src/
│   ├── main.py
│   ├── models.py
│   ├── stability.py
│   └── synthetic_data.py
│
├── README.md
└── requirements.txt
```

---

## Main Findings

The experiments show that:

- Moderate L2 regularization generally improves generalization performance.
- Ridge regression produces more stable models than ordinary least squares.
- Excessive regularization increases bias and eventually degrades predictive performance.
- The relationship between stability and generalization is clearer in the synthetic dataset than in the real-world dataset due to the additional complexity and noise present in real data.

---

## Requirements

Python 3.11+

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python src/main.py
```

Running the script automatically:

- generates the synthetic dataset,
- preprocesses the Communities and Crime dataset,
- trains OLS and Ridge models,
- computes stability metrics,
- produces plots,
- exports the experimental results.

---


