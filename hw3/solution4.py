## Exercise 4 (10 minutes): Ridge vs. Lasso
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset
np.random.seed(42)
num_samples = 30

# Features:
# X1, X2 might be correlated;
# X3 might be less relevant
X1 = np.random.rand(num_samples) * 10
X2 = X1 + np.random.rand(num_samples) * 2  # Correlated with X1
X3 = np.random.rand(num_samples) * 10      # Possibly less relevant

# Define a "true" relationship for the target:
# y = 3*X1 + 1.5*X2 + noise
# X3 might not affect y much
y = 3 * X1 + 1.5 * X2 + np.random.normal(0, 5, size=num_samples)

# Convert to a DataFrame
df = pd.DataFrame({
    "X1": X1,
    "X2": X2,
    "X3": X3,
    "Target": y
})

# 2. Split into features (X) and target (y)
X = df[["X1", "X2", "X3"]]
y = df["Target"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Ridge vs. Lasso
# Train the models
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_coefs = ridge.coef_
ridge_intercept = ridge.intercept_

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_coefs = lasso.coef_
lasso_intercept = lasso.intercept_

# 5. Evaluate on the test set
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

# Calculate R², MSE, and MAE for both models
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

# 6. Compare coefficients and performance
print("True Relationship: y = 3*X1 + 1.5*X2 + noise")
print("\nRidge Coefficients:", ridge.coef_)
print("Ridge Intercept:", ridge.intercept_)
print(f"Ridge R²: {r2_ridge:.3f}, MSE: {mse_ridge:.3f}, MAE: {mae_ridge:.3f}")

print("\nLasso Coefficients:", lasso.coef_)
print("Lasso Intercept:", lasso.intercept_)
print(f"Lasso R²: {r2_lasso:.3f}, MSE: {mse_lasso:.3f}, MAE: {mae_lasso:.3f}")

# Lasso are un R² puțin mai mare decât Ridge, ceea ce sugerează că modelul Lasso explică mai bine variația din date și poate oferi o precizie mai bună în predicții.
# De asemenea, Lasso are un MSE mai mic decât Ridge, ceea ce înseamnă că predicțiile sale sunt mai aproape de valorile reale.
# În cazul Lasso, această regularizare L1 are efectul suplimentar de a aduce coeficienți la zero, ceea ce duce la eliminarea unor caracteristici nerelevante pentru model.
# Astfel, Lasso realizează o selecție implicită a caracteristicilor, păstrând doar acele caracteristici care contribuie semnificativ la prezicerea țintei.