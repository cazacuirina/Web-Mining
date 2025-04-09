## Exercise 6 (10 minutes): Customer Segmentation Use Case
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Generate synthetic customer data
#    - 'purchase_frequency': how many purchases per month
#    - 'average_spent': average amount spent per purchase
#    - 'loyalty_score': a simple 1–5 rating

np.random.seed(42)
num_customers = 50

df_customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 15, num_customers),
    'average_spent': np.random.randint(10, 500, num_customers),
    'loyalty_score': np.random.randint(1, 6, num_customers)
})

print("=== Raw Customer Data (first 5 rows) ===")
print(df_customers.head(), "\n")

# 2. Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_customers)

# 3. K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(df_scaled)

# 4. Add cluster labels to the DataFrame
df_customers['segment'] = cluster_labels

# 5. Inspect each segment
print("=== Customer Segments Summary ===")
print(df_customers.groupby('segment').mean(numeric_only=True), "\n")

# 6. Optional: Quick interpretation
for i in sorted(df_customers['segment'].unique()):
    print(f"Segment {i}:")
    segment = df_customers[df_customers['segment'] == i]
    avg_freq = segment['purchase_frequency'].mean()
    avg_spent = segment['average_spent'].mean()
    loyalty = segment['loyalty_score'].mean()
    print(f"  - Avg. Frequency: {avg_freq:.1f} purchases/month")
    print(f"  - Avg. Spend: ${avg_spent:.2f}")
    print(f"  - Loyalty Score: {loyalty:.1f}/5\n")
