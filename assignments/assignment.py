# %%
import numpy as np
from typing import Any
from scipy import sparse

# TODO: implement the PCA with numpy
# Note that you are not allowed to use any existing PCA implementation from sklearn or other libraries.
class PrincipalComponentAnalysis:
    def __init__(self, n_components: int) -> None:
        """_summary_

        Parameters
        ----------
        n_components : int
            The number of principal components to be computed. This value should be less than or equal to the number of features in the dataset.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    # TODO: implement the fit method
    def fit(self, X: np.ndarray):
        """
        Fit the model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Center
        self.mean = np.mean(X, axis=0)
        Xc = (X - self.mean)
        # Covariance matrix
        covX = np.dot(Xc.T, Xc) / (Xc.shape[0] - 1)
        # Eigenvalues computation
        eigvals, eigvecs = np.linalg.eig(covX) 
        sort_idx = np.argsort(np.abs(eigvals))[::-1][:self.n_components]
        self.components = eigvecs[:, sort_idx]   

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        Xc = X - self.mean
        return Xc @ self.components


# TODO: implement the LDA with numpy
# Note that you are not allowed to use any existing LDA implementation from sklearn or other libraries.
class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Hint:
        -----
        To implement LDA with numpy, follow these steps:
        1. Compute the mean vectors for each class.
        2. Compute the within-class scatter matrix.
        3. Compute the between-class scatter matrix.
        4. Compute the eigenvectors and corresponding eigenvalues for the scatter matrices.
        5. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix W.
        6. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
        """
        # Class means
        labels = np.unique(y)
        class_mean = []
        for c in labels:
            class_mean.append(np.mean(X[y==c,:], axis=0))
        class_mean = np.array(class_mean)
        self.mean = np.mean(X, axis=0)
        # Between class covariance
        self.mean = np.mean(X, axis=0)
        sparse_between = np.zeros((X.shape[1], X.shape[1]))
        for y in labels:
            Xc = (X[y==c,:] - self.mean[c,:])
            sparse_between += np.outer((Xc-self.mean), (Xc-self.mean).T)
        # Within class covariance
        sparse_within = np.zeros((X.shape[1], X.shape[1]))
        for yc in labels:
            Xc = X[y == yc, :]
            sparse_within += np.cov(Xc.T)     
        # Eigenvalues computation
        eigvals, eigvecs = sparse.linalg.eigs(sparse_between, M=sparse_within, k=self.n_components, which="LM")
        sort_idx = np.argsort(np.abs(eigvals))[::-1][:self.n_components]
        self.components = eigvecs[:, sort_idx]   
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        Xc = X - self.mean
        return Xc @ self.components


# TODO: Generating adversarial examples for PCA.
# We will generate adversarial examples for PCA. The adversarial examples are generated by creating two well-separated clusters in a 2D space. Then, we will apply PCA to the data and check if the clusters are still well-separated in the transformed space.
# Your task is to generate adversarial examples for PCA, in which
# the clusters are well-separated in the original space, but not in the PCA space. The separabilit of the clusters will be measured by the K-means clustering algorithm in the test script.
#
# Hint:
# - You can place the two clusters wherever you want in a 2D space.
# - For example, you can use `np.random.multivariate_normal` to generate the samples in a cluster. Repeat this process for both clusters and concatenate the samples to create a single dataset.
# - You can set any covariance matrix, mean, and number of samples for the clusters.
class AdversarialExamples:
    def __init__(self) -> None:
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        """Generate adversarial examples for PCA

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_features : int
            The number of features.

        Returns
        -------
        X: ndarray of shape (n_samples, n_features)
            Transformed values.

        y: ndarray of shape (n_samples,)
            Cluster IDs. y[i] is the cluster ID of the i-th sample.

        """
        n1 = int(n_samples * 0.5)
        n2 = n_samples - n1
        # Mean
        mean = np.zeros((n_features,))
        mean[1] = n_features
        # Variance
        var = np.zeros((n_features,n_features))
        var[0,0] = n_features*n_features
        # Common distribution
        X1 = np.random.multivariate_normal(mean, var, size=n1)
        X2 = np.random.multivariate_normal(-mean, var, size=n2)
        X = np.concatenate([X1, X2])
        Y = np.array(n1*[0] + n2*[1])
        # Split data along the mean
        indices = np.random.choice(np.arange(n_samples), replace=False, size=(n_samples,))
        return X[indices], Y[indices]  