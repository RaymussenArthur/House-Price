# House Price Prediction

This project builds a regression model to predict house sale prices from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data. The process covers exploratory data analysis (EDA), data preprocessing, model training, and model interpretation using SHAP values.

---

## Features

- **Numeric Features**: `GrLivArea`, `OverallQual`, `GarageCars`, `1stFlrSF`, `YearBuilt`, etc.
- **Categorical Features**: `HouseStyle`, `Foundation`, `GarageType`, etc.

Categorical variables are processed using **OneHotEncoder** to convert categories into binary numerical features. Numeric features are scaled using **StandardScaler`.

---

## 🛠️ Libraries & Tools

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `seaborn`
- `matplotlib`
- `SHAP`
- `joblib`
- `scipy`

---

## Workflow

### 1️⃣ Data Cleaning & Preprocessing

- Handling missing values:
  - Categorical columns: filled with 'None'
  - Numeric columns: filled with median or 0
- Log transformation applied to `SalePrice` to reduce skewness and improve model performance.

### 2️⃣ Exploratory Data Analysis (EDA)

- Distribution plots for target variable (before and after log transform)
- Scatterplots to check correlation between features and sale price
- Skewness check to ensure the target is close to normally distributed
- Log-transform `SalePrice` to fix skewness value

### 3️⃣ Model Building

Models used:
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor

Each model was integrated into a **Pipeline** containing:
- Preprocessor (for scaling numeric features and encoding categoricals)
- Regressor (the model)

### 4️⃣ Hyperparameter Tuning

- `Ridge Regression` tuned using `GridSearchCV` to find the optimal alpha value.
- Cross-validation used to assess generalization performance.

### 5️⃣ Model Evaluation

Metrics:
- **RMSE (Root Mean Squared Error)**
- **R² Score**

Both evaluated on the log-transformed predictions and then converted back to the original scale.

### 6️⃣ Model Interpretation

- **SHAP Summary Plot** visualizes each feature's impact on model predictions.
- High SHAP values indicate features that push predictions higher or lower, with color showing whether the feature value was high or low.

---

## 📈 Best Result

Final model: **Ridge Regression**

- RMSE: 26,165
- R²: 0.9107
- Best Alpha: 10

---

## 📊 Example SHAP Summary Plot

> Visualizing feature importance based on impact on predictions:

![SHAP Summary Plot](images/shap_summary.png)

---

## 📎 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/RaymussenArthur/House-Price.git
