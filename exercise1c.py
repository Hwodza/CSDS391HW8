import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define theta range
theta = np.linspace(0, 1, 100)

# Prior parameters for beta distribution (starting with uniform prior)
a_prior, b_prior = 1, 1

# Sequence of coin flips: head, head, tail, head
flips = ["head", "head", "tail", "head"]

# Track number of heads and total flips
y = 0  # heads
n = 0  # total flips

# Perform Bayesian updating after each flip
for i, flip in enumerate(flips):
    # Update count based on the outcome of the current flip
    n += 1
    if flip == "head":
        y += 1

    # Posterior distribution for θ after observing y heads and n flips (Beta distribution)
    a_post, b_post = a_prior + y, b_prior + (n - y)
    posterior = beta.pdf(theta, a_post, b_post)

    # Create a separate figure for each flip
    plt.figure(figsize=(6, 4))
    plt.plot(theta, posterior, label=f'After {n} flip(s): {flip}')
    plt.fill_between(theta, posterior, alpha=0.2)
    plt.title(f'Posterior Distribution After Flip {n}: {flip}')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta|y,n)$')
    plt.legend()

    # Save each plot as a separate image file
    plt.savefig(f'posterior_after_flip_{n}.png')
    plt.close()  # Close the figure to avoid overlap in the loop
