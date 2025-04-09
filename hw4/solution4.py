## Exercise 4 (10 minutes): Agglomerative Clustering & Dendrogram
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we simulate df_scaled by loading and scaling the Iris dataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# ------------------------------------------------------------------------------------

# 2. Perform Agglomerative Clustering
agglomerative_cls = AgglomerativeClustering(n_clusters=3, linkage='ward')

# 3. Add the cluster labels to the DataFrame
df_scaled['cluster'] = agglomerative_cls.fit_predict(df_scaled)

# 4. Print a quick summary of how many points were assigned to each cluster
print("Cluster distribution:")
print(df_scaled['cluster'].value_counts().sort_index())

# 5. Create a linkage matrix for plotting a dendrogram
#    Note: We exclude the 'cluster' column when computing the linkage
Z = linkage(df_scaled.drop(columns='cluster'), method='ward')

# 6. Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           color_threshold=10)
plt.title("Hierarchical Clustering Dendrogram (Iris dataset)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
