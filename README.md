
# Regularization, Stability, and Generalization in Ridge Regression

## Overview

This project studies the relationship between **regularization**, **algorithmic stability**, and **generalization** in supervised learning. The analysis is based on ridge regression implemented from scratch using NumPy.

The main goal is to empirically investigate how the regularization parameter affects:
- training error
- test error
- stability of the learning algorithm

---

## Dataset

The experiments are conducted using the **Communities and Crime dataset** from the UCI Machine Learning Repository.

- Observations: 1994
- Features: ~100 (after preprocessing)

### Preprocessing steps:
- Removed non-predictive identifier columns
- Dropped columns with more than 50% missing values
- Replaced remaining missing values using mean imputation
- Standardized all features using training set statistics

The target variable represents the **violent crime rate per population**, treated as a continuous variable.

---

## Methods

### Models
- Ordinary Least Squares (OLS)
- Ridge Regression

Both models are implemented from scratch.

---

### Stability Measure

Stability is estimated using a leave-one-out approach:
- Remove one training point
- Retrain the model
- Measure the change in predictions on a fixed test set

The stability metric is defined as:
- **Average absolute change in predictions**

---

## Experiments

### 1. Regularization vs Performance
- Train models for different values of α
- Measure:
  - Training MSE
  - Test MSE

### 2. Regularization vs Stability
- Compute stability for each α
- Plot stability against regularization strength

### 3. Stability vs Generalization
- Analyze the relationship between stability and test error
- Compute correlation between the two

### 4. Dataset Size Extension
- Fix α
- Vary training dataset size
- Measure stability and test error

---

## Results

Key findings:

- Increasing regularization improves stability
- Moderate regularization improves generalization
- Excessive regularization leads to underfitting
- More stable models tend to have lower test error
- Larger datasets improve both stability and generalization

---

## Project Structure

```text
regularization-stability-ridge/
│
├── data/
│   └── communities.data
│
├── src/
│   ├── main.py
│   ├── models.py
│   └── stability.py
│
├── results/
│   ├── communities_error_results.csv
│   ├── communities_stability_results.csv
│   ├── communities_results_with_stability.csv
│   ├── communities_dataset_size_results.csv
│   └── plots_communities/
│       ├── error_vs_alpha.png
│       ├── test_error_vs_alpha.png
│       ├── stability_vs_alpha.png
│       ├── stability_vs_test_error.png
│       ├── stability_vs_dataset_size.png
│       └── test_error_vs_dataset_size.png
│
├── README.md
├── requirements.txt
└── report.pdf

```

---
---

## Requirements

The project uses the following Python libraries:

```text
numpy>=1.24
pandas>=2.0
matplotlib>=3.7

```

## How To Run 


1. Clone the repository:

```bash
git clone https://github.com/rhodine543/regularization-stability-ridge.git
```

2. Navigate into the project directory:

```bash
cd regularization-stability-ridge
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Run the main script:

```bash
python src/main.py
```

The script will:
- Train OLS and Ridge regression models
- Compute training and test error
- Estimate algorithmic stability
- Generate plots and CSV result files

All outputs are saved in the `results/` directory.









