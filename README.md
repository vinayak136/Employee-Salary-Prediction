# 💼 Employee Salary Prediction

> Accurately forecasting employee compensation using Machine Learning and Deep Learning — combining Linear Regression, Decision Trees, and a Keras Neural Network into a full-stack salary benchmarking system.

---

## 📌 Project Overview

Employee salary prediction is a critical challenge in HR analytics and compensation management. Guessing salaries leads to underpaying talent or overspending on compensation budgets. This project builds a data-driven solution that predicts an employee's expected salary based on key factors — **experience, education, job role, department, age, and performance score** — using three different modelling approaches and comparing them rigorously.

The project also includes a **Salary Benchmarking Tool** that takes any employee profile as input and returns predicted salaries from all three models, enabling HR teams to make equitable, evidence-based compensation decisions.

---

## 🎯 Objectives

- Predict employee salaries accurately using multiple ML/DL approaches
- Compare Linear Regression, Decision Trees, and Neural Networks on the same dataset
- Identify the most influential features that drive salary differences
- Build a reusable salary benchmarking function for HR decision-making
- Minimise guesswork in compensation planning with quantitative predictions

---

## 🧰 Technologies & Tools

| Category | Tools Used |
|---|---|
| Language | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Deep Learning | TensorFlow 2.x / Keras |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## 📁 Repository Structure

```
Employee-Salary-Prediction/
│
├── Employee_Salary_Prediction.ipynb   ← Main notebook (all models + visualisations)
├── requirements.txt                   ← Python dependencies
└── README.md                          ← This file
```

---

## 📊 Dataset

A realistic synthetic dataset of **2,000 employees** was generated with the following features:

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Employee age (22–65 years) |
| Experience | Numerical | Years of professional experience |
| Education | Categorical | High School / Bachelor / Master / PhD |
| Job Role | Categorical | Analyst / Engineer / Manager / Director / VP / Executive |
| Department | Categorical | IT / Finance / HR / Marketing / Operations / Sales |
| Performance Score | Numerical | Annual performance rating (1.0–5.0) |
| **Salary** | **Target** | **Annual salary in USD** |

Salary is computed with realistic business logic: each job role has a base salary, education adds a bonus, experience contributes linearly, and department applies a multiplier — with random noise added to simulate real-world variance.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Salary distribution histogram
- Experience vs Salary scatter plot
- Average salary by job role, education level, and department
- Correlation heatmap of numerical features

### 2. Data Preprocessing
- Label encoding for categorical variables (Education, Job Role, Department)
- 80/20 train-test split with `random_state=42`
- StandardScaler normalisation (applied for Linear Regression and Neural Network)

### 3. Models Built

#### Model 1 — Linear Regression
- Baseline model to establish a performance floor
- Feature coefficient plot to interpret which factors drive salary up or down
- Diagnostics: Actual vs Predicted, Residual plot, Residual distribution

#### Model 2 — Decision Tree Regressor
- `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=5`
- Feature importance bar chart
- Bias-variance tradeoff analysis: R² plotted across depths 2–15

#### Model 3 — Neural Network (TensorFlow/Keras)
- Architecture: `256 → 128 → 64 → 32 → 1` with BatchNormalization + Dropout
- Optimizer: Adam (lr=0.001)
- Callbacks: EarlyStopping + ReduceLROnPlateau
- Training curves: Loss (MSE) and MAE per epoch

---

## 📈 Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | ~0.87 | ~$17,000 | ~$13,000 |
| Decision Tree | ~0.93 | ~$12,000 | ~$9,000 |
| **Neural Network** | **~0.95** | **~$10,000** | **~$7,000** |

The Neural Network achieves the best generalisation performance with R² ≈ 0.95, meaning it explains 95% of the variance in employee salaries.

---

## 🔍 Key Findings

- **Job Role** is the single most influential predictor of salary — a Director earns roughly 2.5× an Analyst baseline
- **Experience** contributes ~$2,500 per year of additional seniority
- **Education** adds a significant bonus, with PhD holders earning ~$30,000 more than high school graduates at equivalent roles
- **Department** introduces a multiplier effect — IT and Finance employees earn 12–15% more than HR for the same role
- **Performance Score** influences salary but with lower weight than structural factors like role and education

---

## 🛠 Salary Benchmarking Tool

The notebook includes a `predict_salary()` function that accepts any employee profile and returns salary predictions from all three models plus an ensemble average:

```python
predict_salary(
    age=35,
    experience=10,
    education='Master',
    job_role='Manager',
    department='Finance',
    performance_score=4.2
)
# Returns predictions from LR, DT, NN, and Ensemble Average
```

This is directly applicable for HR compensation benchmarking and equitable pay planning.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook Employee_Salary_Prediction.ipynb
```

Run all cells top to bottom. The notebook is self-contained — the dataset is generated in the first cells, no external CSV needed.

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

---

## 🤝 Author

**Vinayak**  
B.Tech CSE | VIT-AP University 

