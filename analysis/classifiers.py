import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as OrigLogisticRegression


class PCAClassifier(BaseEstimator, ClassifierMixin):
    """
    PCA-based binary classifier with data augmentation by +/- (X1 - X0).

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

    def __init__(self):
        pass

    def fit(self, X, y, **fit_params):
        """
        Fit the PCA-based classifier on augmented differences.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary labels (0 or 1). Must have an equal number of 0s and 1s.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        normalize = fit_params.pop('normalize')
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.pca_mean = np.mean(X, axis=0)
        X = X - self.pca_mean

        # Separate the data into the two classes
        X0 = X[y == 0]
        X1 = X[y == 1]
        if len(X0) != len(X1):
            raise ValueError("Number of class-0 and class-1 samples must be the same.")

        # Compute the differences (X1_i - X0_i)
        dX = X1 - X0  # shape: (n_class_samples, n_features)

        # Augment with -dX
        dX_aug = np.vstack([dX, -dX])  # shape: (2*n_class_samples, n_features)

        # Perform PCA on the augmented difference matrix
        pca = PCA(n_components=1)  # Only need the first principal component
        pca.fit(dX_aug)

        # The first principal component
        self.axis_ = pca.components_[0]

        # Flip sign if needed so that the average difference projects positively
        mean_diff = np.mean(dX, axis=0)  # average difference X1 - X0
        if np.dot(mean_diff, self.axis_) < 0:
            self.axis_ = -self.axis_

        self.score_std_ = np.std(np.dot(X, self.axis_)) if normalize else 1.0
        # self.threshold_ = 0
        # return self
        # ---------------------------------------------------------------------
        # Find the best threshold on the training data
        # ---------------------------------------------------------------------
        # For each X0, X1 in training, compute the projection s = w^T x
        proj_0 = X0.dot(self.axis_)
        proj_1 = X1.dot(self.axis_)

        # We want to find threshold b that best separates proj_0 and proj_1
        # i.e., class 0 are "scores <= b", class 1 are "scores > b"
        # We'll do a simple linear scan over candidate thresholds.

        # Combine all projections and labels
        all_scores = np.concatenate([proj_0, proj_1])
        all_labels = np.concatenate([np.zeros_like(proj_0), np.ones_like(proj_1)])

        # Sort by score
        sort_idx = np.argsort(all_scores)
        all_scores_sorted = all_scores[sort_idx]
        all_labels_sorted = all_labels[sort_idx]

        # The best threshold is a midpoint between consecutive scores
        # We'll also consider thresholds below min and above max if needed
        # But typically, a midpoint approach is standard.

        n = len(all_scores_sorted)

        # We track best accuracy and best threshold
        best_acc = -1
        best_thresh = 0

        # We'll do an efficient approach by scanning from left to right
        # For a threshold T, predictions = 1 if score > T, else 0.
        # Let's keep track how many 1's to the right, 0's to the left, etc.

        # Initially, if threshold < all_scores_sorted[0], everything is predicted 1
        # We'll count how many of the true labels are 1 in all_labels
        # Then update as we move the threshold from one point to the next
        num_ones = np.sum(all_labels_sorted == 1)
        num_zeros = n - num_ones

        # Start threshold so low that everything is predicted class 1
        # True positives: all 1's
        # True negatives: 0
        # So initial accuracy = number_of_1 / n
        true_ones_so_far = 0  # how many 1's to the left
        true_zeros_so_far = 0  # how many 0's to the left

        # Compute initial accuracy
        # all predicted = 1 -> correct are only the ones that are truly class 1
        best_acc = np.sum(all_labels_sorted == 1) / n
        best_thresh = all_scores_sorted[0] - 1e-9  # something below the minimum

        # Now iterate through all possible boundaries between points
        for i in range(n):
            # current score
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
        """
        Compute the signed distance (projection) from each sample to the decision axis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Positive if the sample is more like class 1 (above threshold),
            negative if more like class 0 (below threshold).
        """
        check_is_fitted(self, ['axis_', 'threshold_'])
        X = check_array(X)
        return (np.dot(X - self.pca_mean, self.axis_) - self.threshold_) / self.score_std_

    def predict(self, X):
        """
        Predict binary labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        scores = self.decision_function(X)
        return (scores > 0).astype(int)

    def score(self, X, y):
        """
        Returns the average accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)



class PCAScorer(BaseEstimator, ClassifierMixin):
    """
    PCA-based scorer
    """

    def __init__(self):
        pass

    @property
    def axis_(self):
        return self.components_[self.pc_number - 1]

    @property
    def score_std_(self):
        return self.score_std_list[self.pc_number - 1] if self.normalize else 1.0

    def fit(self, X, y, **fit_params):
        """
        Fit the PCA-based classifier on augmented differences.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary labels (0 or 1). Must have an equal number of 0s and 1s.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.normalize = fit_params.pop('normalize')
        self.pc_number = fit_params.pop('pc_number')
        assert self.pc_number >= 1
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.pca_mean = np.mean(X, axis=0)
        X = X - self.pca_mean

        # Perform PCA on the augmented difference matrix
        pca = PCA(n_components=min(X.shape[0], X.shape[1])-1)
        pca.fit(X)

        # The first principal component
        self.components_ = pca.components_

        self.score_std_list = [
            np.std(np.dot(X, self.components_[i])) for i in range(len(self.components_))
        ]

        self.threshold_ = 0
        return self

    def decision_function(self, X):
        """
        Compute the signed distance (projection) from each sample to the decision axis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Positive if the sample is more like class 1 (above threshold),
            negative if more like class 0 (below threshold).
        """
        check_is_fitted(self, ['axis_', 'threshold_'])
        X = check_array(X)
        return (np.dot(X - self.pca_mean, self.axis_) - self.threshold_) / self.score_std_

    def predict(self, X):
        """
        Predict binary labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        scores = self.decision_function(X)
        return (scores > 0).astype(int)

    def score(self, X, y):
        """
        Returns the average accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class LogisticRegression(OrigLogisticRegression):
    """
    LogisticRegression wrapper that rescales decision_function by
    the std of the training decision scores.
    """
    def fit(self, X, y, **fit_params):
        normalize = fit_params.pop('normalize')
        super().fit(X, y, **fit_params)
        # Compute the decision function on the training set
        train_scores = super().decision_function(X)
        self.score_std_ = np.std(train_scores) if normalize else 1.0
        return self

    def decision_function(self, X):
        scores = super().decision_function(X)
        return scores / self.score_std_
