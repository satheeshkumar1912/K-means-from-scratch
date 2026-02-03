import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from kmeans_custom import CustomKMeans

# Synthetic Data Generation
X, _ = make_blobs(
    n_samples=200,
    centers=4,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
df.to_csv("data/synthetic_customer_data.csv", index=False)

# Step 1: Load dataset
data = pd.read_csv("data/synthetic_customer_data.csv")
X = data.values

# Step 2: Select numerical columns only
X = data.select_dtypes(include=["number"]).values

# Step 3: Run Custom K-Means
custom_kmeans = CustomKMeans(n_clusters=4)
custom_kmeans.fit(X)

print("Custom K-Means Inertia:", custom_kmeans.inertia_)

# Plot Custom K-Means result
plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels)
plt.scatter(
    custom_kmeans.centroids[:, 0],
    custom_kmeans.centroids[:, 1],
    color="red",
    marker="X",
    s=200
)
plt.title("Custom K-Means Clustering")
plt.savefig("plots/custom_kmeans.png")
plt.show()

# Step 4: Run Scikit-learn KMeans

sklearn_kmeans = KMeans(n_clusters=4, random_state=42)
sklearn_kmeans.fit(X)

print("Scikit-learn K-Means Inertia:", sklearn_kmeans.inertia_)

# Plot Scikit-learn KMeans result
plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_)
plt.scatter(
    sklearn_kmeans.cluster_centers_[:, 0],
    sklearn_kmeans.cluster_centers_[:, 1],
    color="red",
    marker="X",
    s=200
)
plt.title("Scikit-learn K-Means Clustering")
plt.savefig("plots/sklearn_kmeans.png")
plt.show()
