# Necessary Imports:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def LogisticRegression_func(dataset_type, X_train, X_test, y_train, y_test):    

    logistic_regression = LogisticRegression(max_iter=250)

    #Perform grid search for C(regularization parameter)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 
    grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_['C']
    print("Best value of C:", best_C)

    # Train the final model using the best value of C
    best_logistic_regression = LogisticRegression(C=best_C, max_iter=250 , solver='lbfgs')  # Specify other parameters as needed
    best_logistic_regression.fit(X_train, y_train)

    # Predictions on the test set
    predictions = best_logistic_regression.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Maximum accuracy obtained for dataset type = "+dataset_type+":",accuracy)

    # Plotting C vs. Accuracy
    mean_scores = grid_search.cv_results_['mean_test_score']
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    plt.figure(figsize=(6, 4))
    plt.plot(C_values, mean_scores, marker='o', linestyle='-')
    plt.title('Mean Cross-Validated Accuracy vs. Regularization Parameter (C)')
    plt.xlabel('Regularization Parameter (C)')
    plt.ylabel('Mean Accuracy')
    plt.xscale('log')  # Using log scale for better visualization
    plt.grid(True)
    plt.show()

    return accuracy