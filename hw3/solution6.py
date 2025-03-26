## Exercise 6 (10 minutes): kNN for Regression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset
np.random.seed(42)
num_samples = 30

# Let's generate two features (e.g., Feature1, Feature2) and a target
X = np.random.rand(num_samples, 2) * 10
# Define a "true" relationship for the target: y = 3*X1 + 2*X2 + noise
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)

# Convert to a DataFrame for clarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Feature scaling (recommended for distance-based methods like kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create and train the kNN Regressor for different values of k
k_values = [1, 3, 5, 7]  # Different values for k
r2_scores = []
mse_scores = []
mae_scores = []

# Loop over different values of k
for k in k_values:
    # Create the kNN regressor model
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(X_train_scaled, y_train)  # Train the model on the training set

    # 6. Evaluate the model on the test set
    y_pred = knn_reg.predict(X_test_scaled)

    # Performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Store the metrics for each k
    r2_scores.append(r2)
    mse_scores.append(mse)
    mae_scores.append(mae)

    # Print the performance metrics for each k
    print(f"For k={k}:")
    print(f"  R²: {r2:.3f}")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}\n")

# 7. (Optional) Explore the effect of different k values
# Identify the best k based on R² score
best_k = k_values[np.argmax(r2_scores)]
print(f"Best K: {best_k} with R²={r2_scores[np.argmax(r2_scores)]:.3f}")