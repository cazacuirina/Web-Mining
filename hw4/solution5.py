## Exercise 5 (10 minutes): Evaluating Clusters with Silhouette Scores
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load or assume you have a preprocessed dataset (df_scaled)
#    For demonstration, we'll again load & scale the Iris dataset
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit each clustering method
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Get the cluster labels from each method
kmeans_labels = kmeans.labels_
dbscan_labels = dbscan.labels_
agg_labels = agg.labels_

# 4. Compute silhouette scores (only if more than one cluster exists)
def safe_silhouette_score(X, labels, method_name):
    unique_labels = set(labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(X):
        score = silhouette_score(X, labels)
        print(f"{method_name} Silhouette Score: {score:.3f}")
    else:
        print(f"{method_name} Silhouette Score: Not applicable (only one cluster or too many unique labels)")

# 5. Print the scores
safe_silhouette_score(X_scaled, kmeans_labels, "KMeans")
safe_silhouette_score(X_scaled, dbscan_labels, "DBSCAN")
safe_silhouette_score(X_scaled, agg_labels, "Agglomerative Clustering")
