import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def initialize_centroids(X, K):
    """Randomly initialize centroids from the data points."""
    indices = np.random.choice(X.shape[0], K, replace=False)
    return X[indices]


def compute_distances(X, centroids):
    """Compute the distance from each point to each centroid."""
    distances = np.zeros((X.shape[0], len(centroids)))
    for k, centroid in enumerate(centroids):
        distances[:, k] = np.linalg.norm(X - centroid, axis=1)
    return distances


def assign_clusters(distances):
    """Assign each point to the nearest centroid."""
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, K):
    """Update centroids by computing the mean of points assigned to each cluster."""
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            centroids[k] = np.mean(cluster_points, axis=0)
    return centroids

def compute_objective(X, centroids, labels):
    """Compute the k-means objective function (distortion) for the current clusters."""
    objective_value = 0.0
    for k, centroid in enumerate(centroids):
        cluster_points = X[labels == k]
        objective_value += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
    return objective_value


def kmeans(X, K, max_iters=100, tolerance=1e-4):
    """K-means clustering algorithm."""
    # Step 1: Initialize centroids
    centroids = initialize_centroids(X, K)
    print(centroids)
    for i in range(max_iters):
        # Step 2: Compute distances and assign clusters
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        
        # Step 3: Update centroids
        new_centroids = update_centroids(X, labels, K)
        
        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids
    
    return labels, centroids


def main():
    # Run K-means on the iris dataset with K=3 (since there are 3 species in the dataset)
    # Load the iris dataset from a CSV file
    # Replace 'iris.csv' with the path to your CSV file
    default_filename = 'irisdata.csv'
    filename = input(f"Enter the name of the file (default {default_filename}): ") or default_filename
    data = pd.read_csv(filename, header=1)
    K = int(input("Enter the number of clusters (default 3): ")) or 3
    # Extract feature values (assuming the last column is the label, exclude it if needed)
    # If there is no label column, use data.values directly.
    X = data.iloc[:, :-1].values  # Exclude the last column if it contains labels
    K = 3
    labels, centroids = kmeans(X, K)

    # Visualize the results in 2D (using the first two features of the iris dataset)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel("Feature 1 (e.g., Sepal length)")
    plt.ylabel("Feature 2 (e.g., Sepal width)")
    plt.legend()
    plt.title(f'K-Means Clustering on Iris Dataset (K={K})')
    plt.show()


if __name__ == "__main__":
    main()
