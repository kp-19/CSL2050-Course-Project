# Necessary Imports:
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def Perceptron_model(dataset_type, X_train, X_test, y_train, y_test): 

    perceptron = Perceptron(max_iter=100, random_state=42)
    
    # Perform grid search for learning rate and tolerance values:
    param_grid = {
        'eta0': [0.1, 0.01, 0.001],  # learning rate
        'tol': [1e-2, 1e-3, 1e-4]     # tolerance
    }
    grid_search = GridSearchCV(perceptron, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("For dataset type = "+dataset_type)
    print("Best Parameters:", best_params)

    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return accuracy