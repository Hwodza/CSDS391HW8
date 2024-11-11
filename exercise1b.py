# Re-import necessary libraries for plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Given values for the problem
n = 4  # number of flips
theta = 3 / 4  # probability of heads

# Values of y from 0 to n
y_values = np.arange(0, n + 1)

# Calculate likelihood for each y using the binomial probability mass function
likelihoods = binom.pmf(y_values, n, theta)

# Plot the likelihood as a bar chart with values on top of each bar
plt.figure(figsize=(8, 5))
bars = plt.bar(y_values, likelihoods, color='skyblue', edgecolor='black')

# Annotate each bar with the likelihood value
for bar, likelihood in zip(bars, likelihoods):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{likelihood:.2f}', 
             ha='center', va='bottom')

plt.xlabel("Number of Heads (y)")
plt.ylabel("Likelihood p(y | θ, n)")
plt.title("Likelihood of Getting y Heads for n = 4 and θ = 3/4")
plt.xticks(y_values)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
