# House Price Prediction using Machine Learning
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression Results:")
print(f"MSE: {lr_mse:.4f}")
print(f"R² Score: {lr_r2:.4f}")

# Decision Tree Model
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

print("\nDecision Tree Results:")
print(f"MSE: {dt_mse:.4f}")
print(f"R² Score: {dt_r2:.4f}")

# Feature importance from Decision Tree
print("\nTop 5 Important Features:")
feature_importance = sorted(zip(boston.feature_names, dt_model.feature_importances_), 
                                                       key=lambda x: x[1], reverse=True)[:5]
for feature, importance in feature_importance:
      print(f"{feature}: {importance:.4f}")
