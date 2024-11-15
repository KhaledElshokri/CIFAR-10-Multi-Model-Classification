import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.mean = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.var = np.zeros((len(self.classes), n_features), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            # Adding a small value to variance to avoid division by zero
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def _calculate_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _calculate_posterior(self, x):
        posteriors = []
        for idx, _ in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._calculate_likelihood(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._calculate_posterior(x) for x in X])

# Test the implementation
if __name__ == "__main__":
    # Load features and labels
    train_features = np.load("train_features_pca.npy")
    test_features = np.load("test_features_pca.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    
    # Initialize and train the custom Gaussian Naive Bayes model
    gnb_custom = GaussianNaiveBayes()
    gnb_custom.fit(train_features, train_labels)
    
    # Predict and evaluate
    predictions = gnb_custom.predict(test_features)
    accuracy = np.mean(predictions == test_labels)
    print("Custom Naive Bayes Accuracy:", accuracy * 100)
