"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/16 21:02
Description: 
    

"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate discrete uniform data between 1 and 10
data = np.random.randint(1, 11, size=1000)  # 1-10 inclusive

# Create histogram
plt.hist(data, bins=10, range=(0.5, 10.5), edgecolor='black', color='skyblue', density=True)

# Add theoretical probability density (horizontal line)
x = np.linspace(0.5, 10.5, 100)
y = np.full_like(x, 1/10)  # Probability of each point is 1/10
plt.plot(x, y, 'r-', label='Theoretical Density (1/10)')

# Set title and labels in English
plt.title('Discrete Uniform Distribution (1-10)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()

# Add grid
plt.grid(True, alpha=0.3)

# Show plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for normal distribution
mu = 5.5    # Mean (center of 1-10)
sigma = 1.5 # Standard deviation (adjusted to fit most data in 1-10)

# Generate normal distribution data and clip to 1-10
data = np.random.normal(mu, sigma, size=1000)
data = np.clip(data, 1, 10)  # Restrict values to 1-10

# Create histogram
plt.hist(data, bins=10, range=(1, 10), edgecolor='black', color='skyblue', density=True)

# Add theoretical normal distribution curve
x = np.linspace(1, 10, 100)
pdf = norm.pdf(x, mu, sigma)
# Adjust PDF to account for clipping (normalize within 1-10)
cdf_1 = norm.cdf(1, mu, sigma)
cdf_10 = norm.cdf(10, mu, sigma)
pdf_normalized = pdf / (cdf_10 - cdf_1)  # Normalize to integrate to 1 over [1, 10]
plt.plot(x, pdf_normalized, 'r-', label='theoretical density')

# Set title and labels in English
plt.title('Normal Distribution (1-10)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()

# Add grid
plt.grid(True, alpha=0.3)

# Show plot
plt.show()