import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# Step 1: Load Dataset
# -----------------------------------------------
df = pd.read_csv("sales_data_sample.csv", encoding="ISO-8859-1")

# Display first few rows
print(df.head())

# -----------------------------------------------
# Step 2: Select Numerical Columns for Clustering
# -----------------------------------------------
data = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES']]

# Handle missing values
data = data.dropna()

# -----------------------------------------------
# Step 3: Feature Scaling
# -----------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------------------------
# Step 4: Find Optimal K using Elbow Method
# -----------------------------------------------
wcss = []   # Within Cluster Sum of Squares

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()

# -----------------------------------------------
# Step 5: Apply K-Means with optimal K (choose based on elbow plot, assume k=3)
# -----------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to original data
data['Cluster'] = clusters

print(data.head())

# -----------------------------------------------
# Step 6: Visualize Clusters
# -----------------------------------------------
plt.scatter(data['QUANTITYORDERED'], data['SALES'], c=data['Cluster'])
plt.xlabel("QUANTITYORDERED")
plt.ylabel("SALES")
plt.title("K-Means Clustering Result")
plt.show()
