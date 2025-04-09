## Exercise 2 (10 minutes): K-Means Clustering
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    (containing numeric, imputed, and scaled features).
#    For demonstration, let's simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris
import numpy as np

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate K-Means with a chosen number of clusters, say 3
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. Fit the model to the data
kmeans.fit(df_scaled)

# 4. Extract cluster labels
labels = kmeans.labels_

# 5. (Optional) Add the cluster labels to the DataFrame
df_scaled["Cluster"] = labels

# 6. Print or visualize the results
print("Cluster labels assigned to each sample:")
print(labels)

# 7. Optional quick visualization (for 2D only)
#    If you'd like a scatter plot, choose two features to plot.
plt.figure(figsize=(8, 5))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=labels, cmap='viridis', s=50)
plt.xlabel(df_scaled.columns[0])
plt.ylabel(df_scaled.columns[1])
plt.title("K-Means Clustering (k=3) - Iris Dataset (first 2 features)")
plt.grid(True)
plt.show()
