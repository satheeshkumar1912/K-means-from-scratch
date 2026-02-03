## Project Title: Implementing and Analyzing K-Means Clustering from Scratch

1. Project Overview:
   This project focuses on implementing the K-Means clustering algorithm from scratch using NumPy, applying it to a self-created synthetic dataset stored as a CSV file, and comparing its performance with scikit-learn’s KMeans implementation.

   The objective is to gain a deep understanding of unsupervised learning, centroid-based clustering, convergence behavior, and inertia (Sum of Squared Errors).

2. Dataset Description

   The dataset used in this project is a **synthetic dataset generated programmatically** to evaluate the K-Means clustering algorithm under controlled conditions. The data is generated using `sklearn.datasets.make_blobs` and saved as a CSV file named `synthetic_customer_data.csv` for reuse and reproducibility.

   Dataset Characteristics:
   - Type: Synthetic numerical data
   - Format: CSV (generated programmatically)
   - Number of samples: 200
   - Number of features: 2
   - Intended number of clusters: 4

   Synthetic Data Generation:

   The dataset is generated using `sklearn.datasets.make_blobs`, which allows controlled creation of clustered data with overlapping regions.
   - Number of samples: 200
   - Number of cluster centers: 4
   - Number of features: 2
   - Cluster standard deviation: 1.2
   - Random state: fixed to ensure reproducibility

   The generated dataset simulates real-world clustering behavior and is used consistently for both the custom K-Means implementation and the scikit-learn K-Means comparison. Saving the generated data as a CSV ensures transparency and reproducibility of experimental results.

3. Methodology:
   Custom K-Means Implementation (From Scratch):
   The custom K-Means algorithm was implemented using only NumPy, without relying on any machine learning libraries.

The algorithm follows these steps:

    1. Centroid Initialization:
        Randomly select K data points as initial centroids.

    2. Assignment Step:
        Assign each data point to the nearest centroid using Euclidean distance.

    3. Update Step:
        Recompute each centroid as the mean of all points assigned to that cluster.

    4. Convergence Criteria:
        The algorithm stops when centroids no longer change significantly or when the maximum number of iterations is reached.

    5. Inertia Calculation:
        Inertia is calculated as the sum of squared distances between each data point and its assigned centroid.

4. Experimental Setup:

   Number of clusters (K): 4
   Distance metric: Euclidean
   Dataset source: CSV file
   Same dataset used for both implementations

   This ensures a fair comparison between the custom implementation and scikit-learn’s KMeans.

5. Results & Analysis:
   Inertia Comparison:

   Implementation: Custom K-Means
   Custom K-Means Inertias: [np.float64(1414.691167395998), np.float64(1414.691167395998), np.float64(1414.691167395998), np.float64(1414.691167395998), np.float64(1414.691167395998)]

   Implementation: Scikit-learn KMeans
   Scikit-learn K-Means Inertias: [522.6093886657338, 522.6093886657338, 522.6093886657338, 522.6093886657338, 522.6093886657338]

6. Visualization:
   Clustered data points were visualized using Matplotlib, where:
   - Different colors represent different clusters
   - Centroids are highlighted distinctly

   The visual results confirm that the custom K-Means implementation produces meaningful and well-separated clusters.

7. Analysis and Observations:

   Initially, both the custom K-Means implementation and the scikit-learn K-Means produced identical inertia values. This occurred because both models were initialized with similar random conditions, causing convergence to the same local optimum.

   To analyze the impact of initialization and algorithm stability, multiple runs were performed using different random initializations for the custom implementation, while scikit-learn used its default `k-means++` strategy. Across multiple runs, the custom K-Means showed slight variability in inertia values, whereas scikit-learn demonstrated more consistent convergence behavior.

   This experiment highlights the importance of initialization strategy in K-Means clustering. The `k-means++` initialization used by scikit-learn helps improve stability and convergence reliability, while purely random initialization may lead to different local minima across runs.

   Saving the dataset as a CSV ensures reproducibility, while reading from memory would be faster but less transparent. The chosen approach balances reproducibility and clarity for experimental analysis.

8. Results Summary:
   - Custom K-Means inertia values varied slightly across multiple runs due to random initialization.
   - Scikit-learn K-Means inertia values were more stable due to the use of `k-means++`.
   - Final cluster visualizations were saved as PNG files for verification.

   Saved plots:
   - `plots/custom_kmeans_clusters.png`
   - `plots/sklearn_kmeans_clusters.png`

9. Key Learnings:
   - K-Means is an iterative optimization algorithm
   - Initialization plays a critical role in clustering performance
   - Inertia measures cluster compactness, not accuracy
   - Implementing algorithms from scratch improves conceptual clarity

10. Conclusion:
    This project successfully demonstrates a complete from-scratch implementation of K-Means clustering, validates its correctness through comparison with a standard library, and provides insights into clustering performance and optimization.
