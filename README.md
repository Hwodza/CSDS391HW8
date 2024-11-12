# CSDS391HW8

This repository contains three exercises related to statistical analysis and clustering algorithms. The exercises are implemented in Python and include visualizations using Matplotlib.

## Files

- `exercise1b.py`: This script calculates and plots the likelihood of getting a certain number of heads in a series of coin flips using the binomial distribution.
- `exercise1c.py`: This script performs Bayesian updating for a series of coin flips and plots the posterior distribution after each flip using the beta distribution.
- `exercise5.py`: This script implements the K-means clustering algorithm and applies it to the Iris dataset. It includes functions for initializing centroids, computing distances, assigning clusters, updating centroids, and computing the objective function. It also includes functions for plotting decision boundaries and visualizing the clustering process.
## Usage
### Exercise 1b
To run the likelihood calculation and plot for coin flips:
```sh
python /filepath/exercise1b.py
```
No input is required and the plots should open on your computer.

### Exercise 1c
To run the Bayesian updating and plot the posterior distributions:
```sh
python /filepath/exercise1c.py
```
No input is required and the plots should open on your computer.

### Exercise 5
To run the K-means clustering on the Iris dataset:
```sh
python /filepath/exercise5.py
```
The script will prompt you to enter the name of the CSV file containing the Iris dataset (default is irisdata.csv) and the number of clusters (default is 3). It will then perform K-means clustering and display the results. Then it will perform the tasks required in parts b-d without any user input other than closing the graphs that appeared previously.

## Functions in exercise5.py
- **initialize_centroids(X, K):** Randomly initialize centroids from the data points.
- **compute_distances(X, centroids):** Compute the distance from each point to each centroid.
- **assign_clusters(distances):** Assign each point to the nearest centroid.
- **update_centroids(X, labels, K):** Update centroids by computing the mean of points assigned to each cluster.
- **compute_objective(X, centroids, labels):** Compute the K-means objective function (distortion) for the current clusters.
- **generalkmeans(X, K, max_iters=100, tolerance=1e-4):** General K-means clustering algorithm.
- **kmeans(X, K, max_iters=100, tolerance=1e-4):** K-means clustering algorithm with objective function tracking.
- **plot_decision_boundaries(X, y, centroids, K, title):** Plot the decision boundaries and data points.

## Dependencies

- numpy
- pandas
- matplotlib
- scipy

You can install the required dependencies using pip:
```sh
pip install numpy pandas matplotlib scipy
```