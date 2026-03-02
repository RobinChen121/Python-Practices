#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:08:52 2026

@author: zhenchen

@Python version: 3.13

@disp:  
    
    
"""

# Import libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler

# Load Iris dataset and standardize
iris = datasets.load_iris()
# standardize
scaler = StandardScaler()
X = scaler.fit_transform(iris.data) # visit the data values by visiting the data attribute
y_true = iris.target  # Original species labels

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(X)

# * Map K-means labels to match true labels (for consistent colors)
labels_mapped = np.zeros_like(y_pred)
# pdb.set_trace()
for i in range(3):
    mask = (y_pred == i)
    labels_mapped[mask] = mode(y_true[mask])[0]

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
centers_2d = pca.transform(kmeans.cluster_centers_)

# Plot side by side: K-means vs True labels with consistent colors
fig, axes = plt.subplots(1, 2, figsize=(14,5))

# Left: K-means clusters (mapped)
# 'viridis' is the name of a predefined colormap in Matplotlib.
# It defines a gradient of colors that Matplotlib uses to map numeric values to colors.
# 'viridis' specifically is a smooth gradient from dark purple → blue → green → yellow.
axes[0].scatter(X_2d[:,0], X_2d[:,1], c=labels_mapped, cmap='viridis', s=50)
axes[0].scatter(centers_2d[:,0], centers_2d[:,1], c='red', s=200, marker='X', label='Centroids')
axes[0].set_title("K-means Clustering (Colors Matched)")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
axes[0].legend()

# Right: Original species
axes[1].scatter(X_2d[:,0], X_2d[:,1], c=y_true, cmap='viridis', s=50)
axes[1].set_title("Original Species")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")

plt.show()