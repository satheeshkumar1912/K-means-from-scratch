import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from kmeans_custom import CustomKMeans


# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)


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


# Load Dataset

data = pd.read_csv("data/synthetic_customer_data.csv")
X = data.values


# Multiple Runs for Stability

custom_inertias = []
sklearn_inertias = []

for run in range(5):
    # Custom K-Means (random initialization)
    custom_kmeans = CustomKMeans(n_clusters=4, max_iters=100)
    custom_kmeans.fit(X)
    custom_inertias.append(custom_kmeans.inertia_)

    # Scikit-learn K-Means (k-means++)
    sklearn_kmeans = KMeans(
        n_clusters=4,
        init="k-means++",
        n_init=10
    )
    sklearn_kmeans.fit(X)
    sklearn_inertias.append(sklearn_kmeans.inertia_)


# Visualization (last run)

# Custom K-Means Plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels)
plt.scatter(
    custom_kmeans.centroids[:, 0],
    custom_kmeans.centroids[:, 1],
    marker="X",
    s=200
)
plt.title("Custom K-Means Clustering")
plt.savefig("plots/custom_kmeans_clusters.png")
plt.close()

# Scikit-learn K-Means Plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_)
plt.scatter(
    sklearn_kmeans.cluster_centers_[:, 0],
    sklearn_kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)
plt.title("Scikit-learn K-Means Clustering")
plt.savefig("plots/sklearn_kmeans_clusters.png")
plt.close()

# Print Results
print("Custom K-Means Inertias:", custom_inertias)
print("Scikit-learn K-Means Inertias:", sklearn_inertias)
