Project Title: Implementing and Analyzing K-Means Clustering from Scratch

1) Project Overview
    This project focuses on implementing the K-Means clustering algorithm from scratch using NumPy, applying it to a self-created synthetic dataset stored as a CSV file, and comparing its performance with scikit-learn’s KMeans implementation.

    The objective is to gain a deep understanding of unsupervised learning, centroid-based clustering, convergence behavior, and inertia (Sum of Squared Errors).

2) Dataset Description

    The dataset used in this project is a synthetic customer dataset created manually and stored as a CSV file named: synthetic_customer_data.csv

    Dataset characteristics:
    Type: Synthetic customer data
    Format: CSV
    Dimensions: Numerical features only
    Number of clusters intended: 4

    The dataset simulates real-world customer behavior and is suitable for clustering analysis.

3) Methodology:
    Custom K-Means Implementation (From Scratch):

    The custom K-Means algorithm was implemented using only NumPy, without relying on any machine learning libraries.

The algorithm follows these steps:

    1. Centroid Initialization:
        Randomly select K data points as initial centroids.

    2. Assignment Step
        Assign each data point to the nearest centroid using Euclidean distance.

    3. Update Step
        Recompute each centroid as the mean of all points assigned to that cluster.

    4. Convergence Criteria
        The algorithm stops when centroids no longer change significantly or when the maximum number of iterations is reached.

    5. Inertia Calculation
        Inertia is calculated as the sum of squared distances between each data point and its assigned centroid.

4) Experimental Setup:

    Number of clusters (K): 4
    Distance metric: Euclidean
    Dataset source: CSV file
    Same dataset used for both implementations

    This ensures a fair comparison between the custom implementation and scikit-learn’s KMeans.

5) Results:
    Inertia Comparison:

    Implementation: Custom K-Means Scikit-learn KMeans
    Final Inertia: 

    Implementation: Scikit-learn KMeans
    Final Inertia: 
	(your output value)
	(your output value)

6) Visualization:
    Clustered data points were visualized using Matplotlib, where:
    - Different colors represent different clusters
    - Centroids are highlighted distinctly

    The visual results confirm that the custom K-Means implementation produces meaningful and well-separated clusters.

7) Analysis & Interpretation:
    The custom implementation successfully converges, demonstrating correct algorithm behavior.
    Scikit-learn’s KMeans typically achieves slightly lower inertia due to optimized initialization techniques.
    Random initialization affects convergence speed and final cluster quality.
    Both implementations show similar clustering patterns visually.

8) Key Learnings:
    - K-Means is an iterative optimization algorithm
    - Initialization plays a critical role in clustering performance
    - Inertia measures cluster compactness, not accuracy
    - Implementing algorithms from scratch improves conceptual clarity

9) Conclusion:
    This project successfully demonstrates a complete from-scratch implementation of K-Means clustering, validates its correctness through comparison with a standard library, and provides insights into clustering performance and optimization.