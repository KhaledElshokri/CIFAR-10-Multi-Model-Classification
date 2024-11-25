import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import csv

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
            # Using a random small smoothing factor
            smoothing = np.random.uniform(1e-7, 1e-6)
            self.var[idx, :] = X_c.var(axis=0) + smoothing
            self.priors[idx] = (X_c.shape[0] + 1) / (float(X.shape[0]) + len(self.classes))

    def _calculate_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var ))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _calculate_posterior(self, x):
        posteriors = []
        for idx, _ in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._calculate_likelihood(idx, x)))
            posterior = prior + likelihood
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
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="macro")
    recall = recall_score(test_labels, predictions, average="macro")

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    
    # Save the confusion matrix to a CSV file
    csv_filename = "cm_custom_naiveb.csv"
    np.savetxt(csv_filename, conf_matrix, delimiter=",", fmt='%d')
    print(f"Confusion matrix saved to {csv_filename}")
