# Kaggle: House Price Prediction

This repository contains a complete machine learning project for predicting house sale prices, based on the [Kaggle "House Prices: Advanced Regression Techniques" competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

The project covers the entire data science workflow:
* **Data Cleaning & Preprocessing:** Handling missing values and feature engineering.
* **Exploratory Data Analysis (EDA):** Visualizing distributions and correlations.
* **Model Training & Tuning:** Building and optimizing multiple regression models.
* **Model Interpretation:** Using SHAP to understand *why* the model makes its predictions.

---

## Final Model Performance

The best-performing model was a **Tuned LightGBM (LGBM) Regressor**.

| Model | R² Score (Train) | R² Score (Test) | Test RMSE |
| :--- | :--- | :--- | :--- |
| **Tuned LightGBM** | 0.9777 | 0.8832 | **$29,933.29** |

*Note: RMSE (Root Mean Squared Error) is in dollars. R² measures the proportion of the variance in the sale price that is predictable from the features.*

---

## Tech Stack

* **Data Analysis & Manipulation:** `pandas`, `numpy`, `scipy`
* **Machine Learning & Modeling:** `scikit-learn` (Ridge, Lasso, RandomForest), `xgboost`, `lightgbm`
* **Visualization & Interpretation:** `matplotlib`, `seaborn`, `SHAP`
* **Model Saving:** `joblib`

---

## Project Workflow

### 1. Data Cleaning & Preprocessing
* **Missing Values:** Categorical `NaN` values were filled with `'None'`, while numeric `NaN` values were imputed using the **median** or `0`.
* **Target Variable:** The `SalePrice` was **log-transformed** (`np.log1p`) to normalize its distribution, which significantly improves regression model performance.
* **Feature Scaling:** Numeric features were scaled using `StandardScaler` and categorical features were encoded using `OneHotEncoder`.

### 2. Exploratory Data Analysis (EDA)
* **Target Distribution:** Analyzed the skewness of `SalePrice` and confirmed the effectiveness of the log transform.
* **Feature Correlation:** Used scatterplots and heatmaps to identify key features (like `GrLivArea` and `OverallQual`) that are highly correlated with the sale price.

### 3. Model Building & Tuning
Several regression algorithms were trained and evaluated using a `Pipeline` to streamline preprocessing:
* Ridge Regression (Tuned with `GridSearchCV`)
* Lasso Regression
* Random Forest Regressor
* XGBoost Regressor
* **LightGBM Regressor (Best Performer)**

### 4. Model Interpretation
> **"Why did the model predict this price?"**

To answer this, a **SHAP (SHapley Additive exPlanations)** summary plot was created. This plot visualizes the impact of each feature on the model's output, showing which features are most important and whether high or low values of that feature increase or decrease the predicted price.

---

## How to Run

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/RaymussenArthur/House-Price.git](https://github.com/RaymussenArthur/House-Price.git)
    cd House-Price
    ```

2.  **Install dependencies:**
    (It's highly recommended to create a `requirements.txt` file from your environment)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open the main notebook to see the full analysis and model training process.
