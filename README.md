# Vectorized Logistic Regression from Scratch 🍷

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

A clean, end-to-end implementation of **Binary Logistic Regression with L2 (Ridge) Regularization** built completely from scratch using **NumPy vectorization** — without relying on Scikit-Learn for modeling, splitting, scaling, or evaluation.

The model is trained and evaluated on the **Wine Quality Dataset** (`WineQT.csv`), classifying red wines into high quality ($> 5$) vs. normal/low quality ($\le 5$) based on their physicochemical properties, achieving a test accuracy of **75.88%**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset Overview](#-dataset-overview)
- [Mathematical Formulation](#-mathematical-formulation)
  - [1. Hypothesis Function](#1-hypothesis-function)
  - [2. Regularized Cost Function](#2-regularized-cost-function)
  - [3. Vectorized Gradient Derivation](#3-vectorized-gradient-derivation)
  - [4. Gradient Descent Update Rule](#4-gradient-descent-update-rule)
- [Workflow & Pipeline](#-workflow--pipeline)
- [Hyperparameters & Results](#-hyperparameters--results)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Technologies Used](#-technologies-used)

---

## 🚀 Overview

Many machine learning projects rely directly on high-level APIs like `sklearn.linear_model.LogisticRegression`. The primary objective of this project is **first-principles understanding**: implementing the core algorithms, vectorization routines, data transformations, and mathematical optimization step-by-step from scratch using pure NumPy and Pandas.

---

## ✨ Key Features

- **Zero Black-Box Estimators**: The core model, gradient descent optimizer, feature scaler, data splitter, and evaluation metrics are written manually in Python & NumPy.
- **Vectorized Computations**: Fully vectorized matrix operations ($Xw + b$, $X^T e$) eliminate slow Python loops during forward and backward passes.
- **L2 Regularization (Weight Decay)**: Integrated regularization parameter ($\lambda$) into both the cost function and gradient calculation to prevent overfitting.
- **Numerical Stability**: Includes numerical clipping ($\epsilon = 10^{-15}$) inside log calculations to prevent `log(0)` / `NaN` errors during extreme predictions.
- **Exploratory Data Analysis (EDA)**: Correlation heatmaps and feature distributions using Seaborn and Matplotlib to guide feature selection.

---

## 📊 Dataset Overview

The dataset used is **[WineQT.csv](WineQT.csv)** (Wine Quality Dataset), which contains physicochemical measurements of red wine variants.

- **Total Samples**: 1,143 records
- **Total Features**: 11 physicochemical measurements + `Id` (dropped)
- **Target Variable**: `quality` (ratings from 3 to 8)

### Target Transformation (Binary Classification)
To convert the multi-class quality rating into a binary classification task:
$$\text{Class} = \begin{cases} 1 & \text{if quality } > 5 \quad (\text{Good Quality}) \\ 0 & \text{if quality } \le 5 \quad (\text{Normal / Low Quality}) \end{cases}$$

### Selected Features
Based on correlation analysis with the target and multicollinearity checks, 8 key features were selected:
1. `fixed acidity`
2. `volatile acidity`
3. `citric acid`
4. `chlorides`
5. `total sulfur dioxide`
6. `density`
7. `sulphates`
8. `alcohol`

---

## 📐 Mathematical Formulation

### 1. Hypothesis Function
The linear combination and sigmoid activation map continuous inputs to a probability $\hat{y} \in (0, 1)$:

$$z = Xw + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
- $X \in \mathbb{R}^{m \times n}$ is the feature matrix ($m$ samples, $n$ features)
- $w \in \mathbb{R}^{n}$ is the weight vector
- $b \in \mathbb{R}$ is the scalar bias term

### 2. Regularized Cost Function
Using Binary Cross-Entropy with an L2 regularization penalty to prevent weight explosion:

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(\hat{y}^{(i)} + \epsilon) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)} + \epsilon) \Big] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

- $\epsilon = 1 \times 10^{-15}$ ensures numerical stability.
- $\lambda$ controls regularization strength (bias-variance trade-off).

### 3. Vectorized Gradient Derivation
Let the prediction error vector be $e = \hat{y} - y \in \mathbb{R}^{m}$.

- **Gradient with respect to bias ($b$)**:
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) = \frac{1}{m} \sum_{i=1}^m e_i$$

- **Gradient with respect to weights ($w$)**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w = \frac{1}{m} X^T e + \frac{\lambda}{m} w$$

### 4. Gradient Descent Update Rule
At each iteration $k$, the parameters are updated in the opposite direction of the gradient:

$$w := w - \alpha \frac{\partial J}{\partial w}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Where $\alpha$ is the learning rate.

---

## 🛠 Workflow & Pipeline

The entire pipeline is documented in **[Model.ipynb](Model.ipynb)**:

```
[WineQT.csv]
      │
      ▼
Data Preprocessing (Drop 'Id', Binarize Target: quality > 5)
      │
      ▼
Exploratory Data Analysis (Correlation Heatmap & Feature Selection)
      │
      ▼
Train / Test Split (Scratch: 80% Train, 20% Test, seed=42)
      │
      ▼
Feature Scaling (Z-Score Standardisation: fit on train, transform both)
      │
      ▼
Vectorized Training (Gradient Descent with L2 Regularization)
      │
      ▼
Inference & Evaluation (Decision threshold 0.5 -> Accuracy calculation)
```

### From-Scratch Helper Functions Implemented

| Function | Purpose |
| :--- | :--- |
| `train_test_split(X, y, test_size, random_state)` | Splits data randomly using NumPy permutation without Scikit-Learn |
| `(X - X_mean) / X_std` | Z-score standardisation computed over training distributions |
| `sigmoid(z)` | Vectorized logistic sigmoid activation function |
| `compute_cost(X, y, w, b, lambda_)` | Regularized binary cross-entropy loss with epsilon guard |
| `compute_gradient(X, y, w, b, lambda_)` | Vectorized matrix calculus for $\partial J/\partial w$ and $\partial J/\partial b$ |
| `gradient_descent(...)` | Batch gradient descent loop tracking cost history |
| `train_model(X_train, y_train, alpha, num_iters, lambda_)` | High-level orchestrator initializing zero-weights and training |
| `predict_model(X_test, w_final, b_final)` | Produces binary class labels using threshold $\hat{y} \ge 0.5$ |
| `check_accuracy(predictions, y_test)` | Computes test classification accuracy as a percentage |

---

## 📈 Hyperparameters & Results

The final model configuration and evaluation metrics:

| Parameter / Metric | Value |
| :--- | :--- |
| **Train/Test Split** | 80% / 20% (`test_size=0.2`, `random_state=42`) |
| **Input Features ($n$)** | 8 physicochemical features |
| **Learning Rate ($\alpha$)** | `0.5` |
| **Number of Iterations** | `8000` |
| **Regularization Parameter ($\lambda$)** | `0.1` |
| **Weight Initialization** | Zeros (`w = np.zeros(n)`, `b = 0.0`) |
| **Decision Threshold** | `0.5` |
| **Final Test Accuracy** | **`75.88%`** |

---

## 📁 Repository Structure

```text
.
├── Model.ipynb        # Main Jupyter notebook containing implementation, EDA, training & evaluation
├── WineQT.csv         # Wine Quality dataset
├── .gitignore         # Ignores virtual environments (venv/)
└── README.md          # Comprehensive project documentation
```

---

## 💻 Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook / JupyterLab or VSCode / Cursor with Jupyter extension

### 1. Clone the repository
```bash
git clone https://github.com/tuhinsuvraroy-tsr/VectorizedLogisticRegressionFromScarch.git
cd VectorizedLogisticRegressionFromScarch
```

### 2. Create and activate a virtual environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install required packages
```bash
pip install numpy pandas matplotlib seaborn ipykernel
```

### 4. Run the notebook
Launch Jupyter and open `Model.ipynb`:
```bash
jupyter notebook Model.ipynb
```
Run all cells sequentially to reproduce the exploratory data analysis, model training, and evaluation results.

---

## 🧰 Technologies Used

- **[Python](https://www.python.org/)**: Core programming language.
- **[NumPy](https://numpy.org/)**: Vectorized linear algebra and numerical computing.
- **[Pandas](https://pandas.pydata.org/)**: Data loading, cleaning, and tabular manipulation.
- **[Matplotlib](https://matplotlib.org/)** & **[Seaborn](https://seaborn.pydata.org/)**: Statistical data visualization and correlation analysis.
- **[Jupyter Notebook](https://jupyter.org/)**: Interactive computing and experimentation environment.

---

## 📝 License

This project is open-source and available for educational and learning purposes.
