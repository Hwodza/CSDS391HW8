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


def generalkmeans(X, K, max_iters=100, tolerance=1e-4):
    """General K-means clustering algorithm for part a."""
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


def kmeans(X, K, max_iters=100, tolerance=1e-4):
    """K-means clustering algorithm with objective function tracking for part b, c and d."""
    centroids = initialize_centroids(X, K)
    objective_values = []  
    centroids_history = [centroids]  
    
    for i in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, K)
        centroids_history.append(new_centroids)
        objective_value = compute_objective(X, centroids, labels)
        objective_values.append(objective_value)
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids
    
    return labels, centroids, objective_values, centroids_history


def plot_decision_boundaries(X, y, centroids, K, title):
    """Plot the decision boundaries and data points."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    distances = compute_distances(grid_points, centroids)
    grid_labels = assign_clusters(distances)
    
    Z = grid_labels.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k', cmap='coolwarm', label='Data points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.grid()


def main():
    #
    # Code for part a
    #
    default_filename = 'irisdata.csv'
    filename = input(f"Enter the name of the file (default {default_filename}): ") or default_filename
    data = pd.read_csv(filename, header=1)
    K = int(input("Enter the number of clusters (default 3): ") or 3)
    X = data.iloc[:, :-1].values
    K = 3
    labels, centroids = generalkmeans(X, K)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel("Feature 1 (e.g., Sepal length)")
    plt.ylabel("Feature 2 (e.g., Sepal width)")
    plt.legend()
    plt.title(f'K-Means Clustering on Iris Dataset (K={K})')
    plt.show()
    plt.close()


    #
    # Code for part b
    #
    data = pd.read_csv('irisdata.csv', header=1)  
    X = data.iloc[:, :-1].values
    K_values = [2, 3]
    learning_curves = {}
    centroids_histories = {}

    for k in K_values:
        labels, centroids, objective_values, centroids_history = kmeans(X, k)
        learning_curves[k] = objective_values
        centroids_histories[k] = centroids_history

    for k in K_values:
        print(learning_curves[k])
        plt.figure(figsize=(8, 5))
        plt.plot(learning_curves[k], label=f'K={k}', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value (Distortion)')
        plt.title(f'Learning Curve for K-Means Clustering on Iris Dataset (K={k})')
        plt.legend()
        plt.grid()
        plt.show()
    

    #
    # Code for part c
    #
    X = data.iloc[:, 2:4].values
    K_values = [2, 3]
    learning_curves = {}
    centroids_histories = {}
    for k in K_values:
        labels, centroids, objective_values, centroids_history = kmeans(X, k)
        learning_curves[k] = objective_values
        centroids_histories[k] = centroids_history

    for k in K_values:
        plt.figure(figsize=(8, 5))
    
        # Plot the data points
        plt.scatter(X[:, 0], X[:, 1], c='lightgray', label='Data points', alpha=0.5)
        
        # Get centroid history
        centroids_history = centroids_histories[k]
        
        # Plot initial, intermediate, and final centroids
        for i, centroids in enumerate(centroids_history):
            if i == 0:
                plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label=f'Initial Centroids (K={k})')
            elif i == len(centroids_history) // 2:
                plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='o', label=f'Intermediate Centroids (K={k})')
            elif i == len(centroids_history) - 1:
                plt.scatter(centroids[:, 0], centroids[:, 1], c='green', marker='*', label=f'Final Centroids (K={k})')

        plt.title(f'Cluster Centers Progress for K={k}')
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.legend()
        plt.grid()
        plt.show()
    

    #
    # Code for part d
    #
    for k in K_values:
        labels, centroids, objective_values, centroids_history = kmeans(X, k)
        plot_decision_boundaries(X, labels, centroids, k, f'Decision Boundaries for K={k}')
        plt.show()


if __name__ == "__main__":
    main()
