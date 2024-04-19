
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib as plt

def RandomForestPlot(dataset_type, X_train, X_test, y_train, y_test):
    # Number of trees for testing
    n_trees_range = [1, 5, 10, 20, 50, 100, 200, 500]

    # List used to store accuracies
    accuracies = []

    # Iterating over each number of trees
    for n_trees in n_trees_range:
        # Initializing and training the Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # Predicting on the test set
        predictions = rf_classifier.predict(X_test)

        # Calculating accuracy and appending it to the list
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    print("Maximum accuracy obtained for dataset type = "+dataset_type+":",max(accuracies))
    # Plotting the graph
    plt.figure(figsize=(6, 4))
    plt.plot(n_trees_range, accuracies, marker='o')
    plt.title('Number of Trees vs. Accuracy for preprocessing = '+ dataset_type)
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.xticks(n_trees_range)
    plt.grid(True)
    plt.show()

    return max(accuracies)