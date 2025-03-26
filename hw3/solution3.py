## Exercise 3 (10 minutes): Regression Trees
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset with multiple features
np.random.seed(42)
num_samples = 30
X = np.random.rand(num_samples, 3) * 10  # e.g., three numeric features

# Let's define a "true" relationship for the target:
# Target = 2*Feature1 + 0.5*Feature2^2 - 3*Feature3 + noise
true_y = 2 * X[:, 0] + 0.5 * (X[:, 1]**2) - 3 * X[:, 2]
noise = np.random.normal(0, 5, size=num_samples)  # Add some noise
y = true_y + noise

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2", "Feature3"]]
y = df["Target"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Decision Tree Regressor
#    You can tune hyperparameters like max_depth to control overfitting
tree_reg = DecisionTreeRegressor(random_state=42, max_depth=3)  # Limit depth to avoid overfitting
tree_reg.fit(X_train, y_train)

# 5. Evaluate on the test set
y_pred = tree_reg.predict(X_test)

# 6. Evaluate R², MSE, and MAE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nEvaluation Metrics for the Decision Tree Regressor:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")

# Optional: Inspect feature importances
print("\nFeature Importances:")
print(tree_reg.feature_importances_)

# Optional: You could visualize the tree with:
# from sklearn.tree import export_graphviz
plt.figure(figsize=(10, 8))
from sklearn.tree import plot_tree
plot_tree(tree_reg, filled=True, feature_names=["Feature1", "Feature2", "Feature3"], rounded=True)
plt.title("Decision Tree Regressor")
plt.show()