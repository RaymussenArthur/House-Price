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
- `lightgbm`

---

## Workflow

### 1. Data Cleaning & Preprocessing

- Handling missing values:
  - Categorical columns: filled with 'None'
  - Numeric columns: filled with median or 0
- Log transformation applied to `SalePrice` to reduce skewness and improve model performance.

### 2. Exploratory Data Analysis (EDA)

- Distribution plots for target variable (before and after log transform)
- Scatterplots to check correlation between features and sale price
- Skewness check to ensure the target is close to normally distributed
- Log-transform `SalePrice` to fix skewness value

### 3. Model Building

Models used:
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor

Each model was integrated into a **Pipeline** containing:
- Preprocessor (for scaling numeric features and encoding categoricals)
- Regressor (the model)

### 4. Hyperparameter Tuning

- `Ridge Regression` tuned using `GridSearchCV` to find the optimal alpha value.
- Cross-validation used to assess generalization performance.

### 5. Model Evaluation

Metrics:
- **RMSE (Root Mean Squared Error)**
- **R² Score**

Both evaluated on the log-transformed predictions and then converted back to the original scale.

### 6. Model Interpretation

- **SHAP Summary Plot** visualizes each feature's impact on model predictions.
- High SHAP values indicate features that push predictions higher or lower, with color showing whether the feature value was high or low.

### 7. Another model training
Training model using Light GBM Regression algorithm. The model result:
- RMSE: 28681.16
- R2 Train: 0.9882
- R2 Test: 0.8928

Tuning model with result:
- Test RMSE: 29933.29
- R2 Train: 0.9777
- R2 Test: 0.8832


---

## 📎 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/RaymussenArthur/House-Price.git
