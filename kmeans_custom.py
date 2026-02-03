import numpy as np

class CustomKMeans:
    def __init__(self, n_clusters=4, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        # Step 1: Initialize centroids randomly from data points
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Step 2: Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Step 3: Update centroids
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if len(cluster_points) == 0:
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))
            new_centroids = np.array(new_centroids)

            # Step 4: Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # Step 5: Calculate inertia (SSE)
        self.inertia_ = np.sum((X - self.centroids[self.labels]) ** 2)
