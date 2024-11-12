import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


theta = np.linspace(0, 1, 100)


a_prior, b_prior = 1, 1


flips = ["head", "head", "tail", "head"]


y = 0  
n = 0  

# Perform Bayesian updating after each flip
for i, flip in enumerate(flips):
    n += 1
    if flip == "head":
        y += 1

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

    plt.savefig(f'posterior_after_flip_{n}.png')
    plt.close()
plt.show()