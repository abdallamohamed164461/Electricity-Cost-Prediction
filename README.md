# ⚡ Electricity Cost Prediction

A regression-based machine learning project that predicts a site's **electricity cost** based on structural, resource-usage, and environmental features. Includes full data analysis, statistical inference, model comparison, and a deployable Streamlit web app.

---

## 📋 Overview

- **Track:** Regression (Track A)
- **Goal:** Predict `electricity cost` for a building/site
- **Approach:** Data cleaning → EDA → statistical analysis → ML modeling → evaluation → deployment

---

## 📊 Dataset

| Property | Value |
|---|---|
| Name | Electricity Cost Prediction Dataset |
| Source | Kaggle |
| Rows | 10,000 |
| Columns | 9 (8 features + 1 target) |
| Target variable | `electricity cost` |
| Missing values | None |

**Numerical features:** `site area`, `water consumption`, `recycling rate`, `utilisation rate`, `air qality index`, `issue reolution time`, `resident count`

**Categorical feature:** `structure type` (Residential, Commercial, Mixed-use, Industrial)

---

## 🔍 Project Steps

1. **Data Loading & Understanding** — inspected shape, types, and summary statistics.
2. **Data Cleaning** — verified no missing values; detected and capped (winsorized) outliers in `water consumption`, `resident count`, and the target using the IQR method.
3. **Exploratory Data Analysis** — 5 visualizations: target distribution, boxplot by structure type, scatterplots, and a correlation heatmap.
4. **Statistical Analysis** — one-way ANOVA test on structure type, OLS regression for coefficient interpretation (standard errors & p-values), a 95% confidence interval for the mean cost, and bootstrap resampling.
5. **Data Preparation** — one-hot encoding of `structure type`, train/test split (80/20), feature scaling with `StandardScaler`.
6. **Modeling** — trained **Linear Regression** and a **KNN Regressor** tuned via `GridSearchCV` (best `k`).
7. **Evaluation** — compared both models on MAE, MSE, RMSE, and R².
8. **Prediction Interval** — bootstrap-based 95% prediction intervals for the final model.
9. **Model Saving** — final model, scaler, and feature list saved with `joblib` for use in the app.

---

## 🤖 Models & Results

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 245.88 | 97,375.95 | 312.05 | 0.9219 |
| **KNN Regressor (k=11)** | **219.69** | **76,986.39** | **277.46** | **0.9383** |

**Best model: KNN Regressor (k=11)** — it achieved the lowest error across all metrics and explains ~93.8% of the variance in electricity cost, outperforming Linear Regression, which suggests the relationship between the features and cost is not purely linear.

### Key Findings
- Structure type has a **statistically significant** effect on electricity cost (ANOVA, p < 0.001).
- `site area` and `resident count` show the strongest visual/correlation relationship with electricity cost.
- The bootstrap-based prediction interval had ~61% empirical coverage on the test set (target: 95%). This is because it captures uncertainty from *model retraining* (bootstrap on the training set) but not the full residual noise in individual predictions — a known limitation worth noting when defending the project, and a good discussion point for future improvement (e.g. combining it with residual-based intervals).

---

## 📁 Files in this Repository

| File | Description |
|---|---|
| `Electricity_Cost_Prediction.ipynb` | Full analysis notebook (cleaning, EDA, stats, ML, evaluation) |
| `electricity_cost_dataset.csv` | Raw dataset |
| `app.py` | Streamlit web application |
| `model.pkl` | Final trained model (KNN Regressor) |
| `scaler.pkl` | Fitted `StandardScaler` used on features |
| `feature_columns.pkl` | Ordered list of feature columns expected by the model |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## ▶️ How to Run Locally

1. **Clone/download this project folder** and open a terminal inside it.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the analysis notebook:**
   ```bash
   jupyter notebook Electricity_Cost_Prediction.ipynb
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser at `http://localhost:8501`.

> Note: `model.pkl`, `scaler.pkl`, and `feature_columns.pkl` are already included, generated at the end of the notebook. If you re-run the notebook, they will be regenerated automatically.

---

## 🚀 Deployment on Streamlit Community Cloud

1. **Create a GitHub repository** and push these files to it:
   `app.py`, `requirements.txt`, `model.pkl`, `scaler.pkl`, `feature_columns.pkl` (the notebook and CSV are optional for deployment but good to include for transparency).

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.

3. Click **"New app"**, then select:
   - Your repository
   - The branch (usually `main`)
   - The main file path: `app.py`

4. Click **"Deploy"**. Streamlit Cloud will automatically install everything in `requirements.txt` and launch the app.

5. Once deployed, you'll get a public URL like:
   `https://your-app-name.streamlit.app`
   which anyone can open from a browser — no installation needed.

6. Any time you push new commits to the connected GitHub repo, the deployed app updates automatically.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `SciPy` · `Matplotlib` · `Seaborn` · `Streamlit` · `joblib`
