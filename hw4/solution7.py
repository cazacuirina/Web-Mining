## Exercise 7 (10 minutes): Anomaly Detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generate synthetic "normal" data
#    E.g., two features representing normal operating ranges (e.g., purchase amounts, usage rates, etc.)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))  # 200 points around mean=50

# 2. Generate synthetic "anomalous" data
#    Points that deviate significantly from the normal distribution
outliers = np.array([[100, 100], [10, 90], [90, 10], [120, 40], [40, 120]])

# 3. Combine the datasets
X = np.vstack((normal_data, outliers))

# 4. Apply DBSCAN
#    eps controls the neighborhood radius; min_samples is how many samples must be within eps to form a cluster
dbscan = DBSCAN(eps=15, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# 5. Identify outliers (DBSCAN labels them as -1)
outlier_indices = np.where(dbscan_labels == -1)[0]

# 6. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50, marker='o')
plt.scatter(X[outlier_indices, 0], X[outlier_indices, 1], c='red', label='Outliers', s=100, edgecolors='black')
plt.title("DBSCAN Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# 7. Reporting
print(f"Total points: {X.shape[0]}")
print(f"Anomalies detected: {len(outlier_indices)}")
