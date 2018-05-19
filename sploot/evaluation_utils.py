"""Utilities to diagnose a model.
"""

import numpy as np
from matplotlib import pyplot as plt

def plot_decision_boundary(model, X, y):
    """Visualize multi-dimensions in 2D, enabling a visual inspection of how
    overfit a model might be.

    Parameters
    ----------
    model: function, a function that can take a feature array and target vector
        as inputs. i.e. LogisticRegression.predict from sklearn is acceptable.
    X: array, (# of samples, # of features) array.
    y: vector, 1D array-like object (# of samples, 1).

    Returns
    -------
    None, plots the decision boundary.
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)