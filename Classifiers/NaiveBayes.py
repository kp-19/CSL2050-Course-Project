# Necessary Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Complement Naive-Bayes(MultinomialNB)
def NaiveBayes(dataset_type, X_train, X_test, y_train, y_test):
    clf = ComplementNB()

    #Perform grid search to get best value of alpha(Laplace Smoothing):
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    best_clf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = best_clf.predict(X_test)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print()
    print("Maximum accuracy obtained for dataset type = "+dataset_type+":",test_accuracy)

    # Plot alpha vs. accuracy
    alphas = [0.1, 0.5, 1.0]
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, mean_test_scores, marker='o')
    plt.title('Alpha vs. Accuracy Score')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Test Accuracy')
    plt.grid(True)
    plt.show()

    return test_accuracy