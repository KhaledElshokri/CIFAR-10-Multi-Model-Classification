from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np

if __name__ == "__main__":
    # Load features and labels
    train_features = np.load("train_features_pca.npy")
    test_features = np.load("test_features_pca.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    
    # Initialize and train the Scikit-Learn Decision Tree model
    tree_sklearn = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=42)
    tree_sklearn.fit(train_features, train_labels)
    
    # Predict and evaluate
    predictions = tree_sklearn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="macro")
    recall = recall_score(test_labels, predictions, average="macro")
    
    print("Scikit-Learn Decision Tree Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Precision:", precision)
    print("Recall:", recall)

    # Save the confusion matrix to a CSV file
    csv_filename = "cm_sklearn_Tree.csv"
    np.savetxt(csv_filename, conf_matrix, delimiter=",", fmt='%d')
    print(f"Confusion matrix saved to {csv_filename}")
