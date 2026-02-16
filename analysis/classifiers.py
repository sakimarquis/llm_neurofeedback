import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as OrigLogisticRegression
from sklearn.covariance import OAS


class PCAClassifier(BaseEstimator, ClassifierMixin):
    """PCA-based binary classifier with data augmentation by +/- (X1 - X0).

    Steps:
      1) Let X0 be samples with y=0 and X1 with y=1 (same number of samples).
      2) Compute dX = (X1 - X0). The augmented data for PCA is { dX, -dX }.
      3) Fit PCA on the augmented data -> get first principal component w.
      4) Determine sign of w so that the average difference is positive:
         if mean( dX @ w ) < 0, flip w := -w.
      5) Find the best threshold b to separate X0 from X1 by maximizing training accuracy.

    The decision function for a new point x is s(x) = w^T x.
    We predict class 1 if s(x) > b, else class 0.
    """
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.pca_mean = None
        self.axis_ = None
        self.threshold_ = None
        self.score_std_ = None

    def fit(self, X, y):
        """Fit a PCA-based binary classifier using paired class-difference augmentation.
        The method constructs paired differences (X1_i - X0_i) between
        class-1 and class-0 samples, augments them with their negatives,
        and applies PCA to extract a single discriminative direction.
        A decision threshold is then selected to maximize training accuracy.

        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples,) Binary labels (0 or 1). Must have an equal number of 0s and 1s.
        :return: self Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.pca_mean = np.mean(X, axis=0)
        X = X - self.pca_mean

        X0 = X[y == 0]
        X1 = X[y == 1]
        if len(X0) != len(X1):
            raise ValueError("Number of class-0 and class-1 samples must be the same.")

        dX = X1 - X0  # shape: (n_class_samples, n_features)
        dX_aug = np.vstack([dX, -dX])  # shape: (2*n_class_samples, n_features)
        pca = PCA(n_components=1)  # Only need the first principal component
        pca.fit(dX_aug)
        self.axis_ = pca.components_[0]

        # Flip sign so that the average difference projects positively
        mean_diff = np.mean(dX, axis=0)
        if np.dot(mean_diff, self.axis_) < 0:
            self.axis_ = -self.axis_

        self.score_std_ = np.std(np.dot(X, self.axis_)) if self.normalize else 1.0

        # For each X0, X1 in training, compute the projection s = w^T x
        proj_0 = X0.dot(self.axis_)
        proj_1 = X1.dot(self.axis_)

        # We want to find threshold b that best separates proj_0 and proj_1
        # i.e., class 0 are "scores <= b", class 1 are "scores > b"
        # We'll do a simple linear scan over candidate thresholds.

        all_scores = np.concatenate([proj_0, proj_1])
        all_labels = np.concatenate([np.zeros_like(proj_0), np.ones_like(proj_1)])
        sort_idx = np.argsort(all_scores)  # Sort by score
        all_scores_sorted = all_scores[sort_idx]
        all_labels_sorted = all_labels[sort_idx]

        # The best threshold is a midpoint between consecutive scores
        # We'll also consider thresholds below min and above max if needed
        # But typically, a midpoint approach is standard.

        n = len(all_scores_sorted)
        # We'll do an efficient approach by scanning from left to right
        # For a threshold T, predictions = 1 if score > T, else 0.
        # Let's keep track how many 1's to the right, 0's to the left, etc.

        # Initially, if threshold < all_scores_sorted[0], everything is predicted 1
        # We'll count how many of the true labels are 1 in all_labels
        # Then update as we move the threshold from one point to the next
        num_ones = np.sum(all_labels_sorted == 1)

        # Start threshold so low that everything is predicted class 1
        # True positives: all 1's, True negatives: 0, So initial accuracy = number_of_1 / n
        true_ones_so_far = 0  # how many 1's to the left
        true_zeros_so_far = 0  # how many 0's to the left

        # Compute initial accuracy
        # all predicted = 1 -> correct are only the ones that are truly class 1
        best_acc = np.sum(all_labels_sorted == 1) / n
        best_thresh = all_scores_sorted[0] - 1e-9  # something below the minimum

        # Now iterate through all possible boundaries between points
        for i in range(n):
            score_i = all_scores_sorted[i]
            label_i = all_labels_sorted[i]

            # "Move" the threshold to the next midpoint
            # Now the sample i that was previously predicted as 1 (since threshold < score_i)
            # will switch side if we set threshold right at score_i
            if label_i == 1:
                true_ones_so_far += 1
            else:
                true_zeros_so_far += 1

            # The threshold after i-th sample is the midpoint between score_i and score_{i+1},
            # or slightly bigger than score_i to separate it from the next.
            if i < n - 1:
                # mid = (score_i + score_{i+1}) / 2.0
                mid = (score_i + all_scores_sorted[i + 1]) * 0.5
            else:
                # after the last sample, threshold is bigger than all scores
                mid = score_i + 1e-9

            # Now let's compute accuracy if we set threshold = mid
            # Points with score <= mid -> predicted 0
            # Points with score > mid  -> predicted 1

            # We know up to index i, these points have scores <= score_i (<= mid).
            # Among these, how many are truly 0? true_zeros_so_far
            # The rest up to i are 1's -> true_ones_so_far
            # The points from i+1 to n-1 have scores > score_i (and so > mid).
            # Among these, how many are truly 1? total_ones - true_ones_so_far
            # Because total_ones = num_ones
            # total_zeros = num_zeros

            correct_0_left = true_zeros_so_far
            correct_1_right = num_ones - true_ones_so_far
            acc = (correct_0_left + correct_1_right) / n

            if acc > best_acc:
                best_acc = acc
                best_thresh = mid

        self.threshold_ = best_thresh
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['axis_', 'threshold_'])
        X = check_array(X)
        return (np.dot(X - self.pca_mean, self.axis_) - self.threshold_) / self.score_std_

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class PCAScorer(BaseEstimator, ClassifierMixin):
    def __init__(self, pc_number, normalize=True):
        self.normalize = normalize
        self.pca_mean = None
        self.components_ = None
        self.threshold_ = 0
        self.score_std_list = None
        self.pc_number = pc_number
        self.d_prime_ = []

    @property
    def axis_(self):
        return self.components_[self.pc_number - 1]

    @property
    def score_std_(self):
        return self.score_std_list[self.pc_number - 1] if self.normalize else 1.0

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.pca_mean = np.mean(X, axis=0)
        X = X - self.pca_mean
        pca = PCA(n_components=min(X.shape[0], X.shape[1]) - 1)
        pca.fit(X)
        self.components_ = pca.components_
        self.score_std_list = np.sqrt(pca.explained_variance_ * (X.shape[0] - 1) / X.shape[0])
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['axis_', 'threshold_'])
        X = check_array(X)
        return (np.dot(X - self.pca_mean, self.axis_) - self.threshold_) / self.score_std_

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def compute_d_prime(self, precision, X):
        scores = np.dot(X - self.pca_mean, self.components_.T)

        for i in range(512):  # only compute d prime for the first 512 columns
            col_scores = scores[:, i]
            mask = col_scores >= self.threshold_
            c1 = X[mask]
            c0 = X[~mask]
            d_prime = compute_highdim_d_prime(c0, c1, precision)
            self.d_prime_.append(d_prime)

        return self.d_prime_


class LogisticRegression(OrigLogisticRegression):
    """LogisticRegression wrapper that rescales decision_function by
    the std of the training decision scores.
    """

    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        super().__init__(**kwargs)
        self.score_std_ = None
        self.d_prime_ = []
        self.axis_ = None

    def fit(self, X, y, **kwargs):
        super().fit(X, y)
        # Compute the decision function on the training set
        train_scores = super().decision_function(X)
        self.score_std_ = np.std(train_scores) if self.normalize else 1.0
        self.axis_ = self.coef_.flatten()
        return self

    def decision_function(self, X):
        scores = super().decision_function(X)
        return scores / self.score_std_

    def compute_d_prime(self, precision, X, y):
        y = np.array(y)
        cluster_0 = X[y == 0]
        cluster_1 = X[y == 1]
        d_prime = compute_highdim_d_prime(cluster_0, cluster_1, precision)
        self.d_prime_ = [d_prime]
        return self.d_prime_


def compute_highdim_d_prime(cluster_0, cluster_1, precision):
    mu_0 = np.mean(cluster_0, axis=0)
    mu_1 = np.mean(cluster_1, axis=0)
    delta = mu_1 - mu_0  # (d,)
    d_squared = delta @ precision @ delta  # (1,)
    return np.sqrt(d_squared)


class OASMahalanobisOperator:
    """Efficient Mahalanobis distance operator under OAS shrinkage.

    Computes delta^T Σ^{-1} delta without forming d×d matrices,
    using Woodbury identity in sample space (n×n).

    Σ = (1 - rho) * S + rho * mu * I
    S = (1 / (n - 1)) * X^T X ,  X is globally centered
    """
    def __init__(self):
        self.X_centered = None          # (n, d) globally centered data, d > n
        self.shrinkage_iso = None       # alpha = rho * mu
        self.shrinkage_scale = None     # beta  = (1 - rho) / (n - 1)
        self.woodbury_cholesky = None   # Cholesky of (I + (beta/alpha) X X^T)

    def fit(self, X: np.ndarray):
        n, d = X.shape
        # Global centering for covariance estimation
        self.X_centered = X_centered = X - X.mean(axis=0, keepdims=True)

        # Estimate OAS shrinkage parameter
        oas = OAS(assume_centered=True).fit(X_centered)
        rho = float(oas.shrinkage_)

        # mu = trace(S) / d = ||X||_F^2 / ((n - 1) * d)
        fro_norm_sq = float(np.sum(X_centered ** 2))
        mu = fro_norm_sq / ((n - 1) * d)

        self.shrinkage_iso = alpha = rho * mu
        self.shrinkage_scale = beta = (1.0 - rho) / (n - 1)

        # Woodbury matrix in sample space: I + (beta / alpha) X X^T
        gram_matrix = X_centered @ X_centered.T
        woodbury_matrix = np.eye(n, dtype=X_centered.dtype) + (beta / alpha) * gram_matrix

        # Pre-factorize once for repeated solves
        self.woodbury_cholesky = np.linalg.cholesky(woodbury_matrix)
        return self

    def mahalanobis_squared(self, mean_diff: np.ndarray) -> float:
        """Compute squared Mahalanobis distance: mean_diff^T Σ^{-1} mean_diff"""
        mean_diff = np.asarray(mean_diff)
        term_iso = (mean_diff @ mean_diff) / self.shrinkage_iso  # isotropic part

        # Woodbury correction in sample space
        projection = self.X_centered @ mean_diff
        z = np.linalg.solve(self.woodbury_cholesky, projection)
        w = np.linalg.solve(self.woodbury_cholesky.T, z)
        term_corr = (self.shrinkage_scale / (self.shrinkage_iso ** 2)) * (projection @ w)
        return term_iso - term_corr
