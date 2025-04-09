## Exercise 3 (10 minutes): DBSCAN Clustering
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we'll again simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate DBSCAN with chosen parameters
#    eps defines the neighborhood radius, min_samples is the minimum number of points
#    for a region to be considered dense.
dbscan = DBSCAN(eps=0.6, min_samples=5)

# 3. Fit the model to the data
dbscan.fit(df_scaled)

# 4. Extract cluster labels
labels = dbscan.labels_

# 5. Identify outliers (DBSCAN labels outliers as -1)
n_outliers = sum(labels == -1)

# 6. (Optional) Add the labels to the DataFrame
df_scaled["Cluster"] = labels

# 7. Print the cluster label counts
print("No. points in cluster:")
print(df_scaled["Cluster"].value_counts().sort_index())
print(f"No. of outliers: {n_outliers}")

# 8. Optional quick visualization (for 2D only)
#    Choose two features to plot, coloring by DBSCAN labels
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1],
                      c=labels, cmap='viridis', s=50)
plt.xlabel(df_scaled.columns[0])
plt.ylabel(df_scaled.columns[1])
plt.title("DBSCAN Clustering - Iris Dataset (first 2 features)")
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()
