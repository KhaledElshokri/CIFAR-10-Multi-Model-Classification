import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

class DecisionTree:
    def __init__(self, max_depth=50, min_samples_split=10, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def gini_index(self, y):
        classes = np.unique(y)
        gini = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            gini -= p ** 2
        return gini

    def split(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] < threshold
        right_indices = ~left_indices
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    def best_split(self, X, y):
        best_gini = float("inf")
        best_feature, best_threshold = None, None
        num_features = X.shape[1]
        features = np.random.choice(num_features, self.max_features, replace=False) if self.max_features else range(num_features)

        for feature_index in features:
            # Use only a few quantiles to reduce the number of thresholds
            thresholds = np.percentile(X[:, feature_index], [25, 50, 75])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                gini = (len(y_left) * self.gini_index(y_left) + len(y_right) * self.gini_index(y_right)) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        num_samples, num_classes = len(y), len(np.unique(y))
        
        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            leaf_value = self.calculate_leaf_value(y)
            return leaf_value

        feature_index, threshold = self.best_split(X, y)
        if feature_index is None:
            return self.calculate_leaf_value(y)

        X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        return (feature_index, threshold, left_subtree, right_subtree)

    def calculate_leaf_value(self, y):
        return np.argmax(np.bincount(y))

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _predict(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_index, threshold, left, right = tree
        if sample[feature_index] < threshold:
            return self._predict(sample, left)
        else:
            return self._predict(sample, right)

# Test the custom Decision Tree
if __name__ == "__main__":
    # Load features and labels
    train_features = np.load("train_features_pca.npy")
    test_features = np.load("test_features_pca.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    
    # Initialize and train the custom Decision Tree model
    tree_custom = DecisionTree(max_depth=20, min_samples_split=10, max_features=10)
    tree_custom.fit(train_features, train_labels)
    
    # Predict and evaluate
    predictions = tree_custom.predict(test_features)
    accuracy = np.mean(predictions == test_labels)
    conf_matrix = confusion_matrix(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="macro")
    recall = recall_score(test_labels, predictions, average="macro")
    
    print("Custom Decision Tree Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)

    # Save the confusion matrix to a CSV file
    csv_filename = "cm_custom_Tree.csv"
    np.savetxt(csv_filename, conf_matrix, delimiter=",", fmt='%d')
    print(f"Confusion matrix saved to {csv_filename}")
