import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Define a complex 2D function with two classes
def complex_func(x1, x2):
    return np.sin(x1) * np.cos(x2) + np.random.randn()

# Generate some example data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.array([0 if complex_func(x1, x2) < 0 else 1 for x1
            , x2 in X])

# Train an SVM model with a non-linear kernel
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)

# Obtain the support vectors and dual coefficients
support_vectors = clf.support_vectors_
dual_coef = clf.dual_coef_

# Calculate the margin using the norm of the support vectors and dual coefficients
margin = 2 / np.sqrt(np.sum(dual_coef ** 2))

# Create a meshgrid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and margin
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f"SVM with {clf.kernel} kernel and margin={margin:.2f}")
plt.show()