# Ecommerce Sales Analytics — Descriptive, Perspective & Predictive Statistics

This repo contains three Jupyter notebooks analyzing the same **Ecommerce_Sales_Dataset.csv** from different statistical angles:

1. **Descriptive Statistics (Midterm)**
2. **Perspective (Optimization) Statistics (Final)**
3. **Predictive Statistics (Final)**

---

## 🗂 Files

| File | Purpose |
|------|---------|
| `Descriptive_statistics_Midterm.ipynb` | Exploratory data analysis and descriptive stats (distributions, correlations, visuals). |
| `Perspective_statistics_Final.ipynb`  | Custom optimization-based regression using NumPy/SciPy (`minimize`, `linprog`) to fit weights and explore “best” parameter sets. |
| `Predictive_statistics_Final.ipynb`   | Machine learning regression pipeline (scikit-learn LinearRegression + feature selection) and a small Dense neural network with TensorFlow/Keras. |
| `Ecommerce_Sales_Dataset.csv`         | Dataset (1,000 rows × 7 cols) with pricing, discount, marketing spend, and units sold. |
| `README.md`                           | This file. |
| `LICENSE`                             | MIT license text. |
| `.gitignore`                          | Ignore temp/runtime artifacts. |
| `requirements.txt` *(optional)*       | Python dependencies. |

---

## 🔍 Notebook Summaries

### 1. Descriptive Statistics (Midterm)
- **Libraries:** pandas, numpy, matplotlib, seaborn, scipy
- **Steps:**
  - Load CSV, check duplicates/missing values
  - Summary stats, z-scores
  - Visuals: histograms, boxplots, pie/bar charts for categorical features
  - Correlation heatmap
- **Goal:** Understand data distribution and relationships before any modeling.

### 2. Perspective Statistics (Final)
- **Libraries:** pandas, numpy, matplotlib, seaborn, `scipy.optimize` (`minimize`, `linprog`)
- **Steps:**
  - Build design matrix `X` manually (with intercept & features)
  - Define MSE loss and a fitness function
  - Use optimization routines to find weight vectors that minimize error (and optionally respect constraints)
- **Goal:** “Prescriptive/Perspective” angle—optimize coefficients directly instead of using off-the-shelf regressors.

### 3. Predictive Statistics (Final)
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, TensorFlow/Keras
- **Steps:**
  - One-hot encode categorical vars, scale numeric features
  - Train/test split
  - **LinearRegression** baseline
  - Feature selection: `SelectKBest`, `SequentialFeatureSelector`
  - Small **Dense neural network** (Keras Sequential) trained for 5 epochs
  - Metrics: R², MSE, MAE; plus visualization (Actual vs Predicted scatter)
  - (Notebook also tries a simple EMA_50 baseline for comparison)
- **Goal:** Build and compare predictive models for **Units_Sold** (continuous target).

---

## ▶️ How to Run

# 1. Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) Create & activate a venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt
# or minimum:
pip install pandas numpy matplotlib seaborn scipy scikit-learn tensorflow

# Open notebooks:

jupyter notebook

