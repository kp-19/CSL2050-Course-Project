## naive-bayes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Create a Complement Naive Bayes classifier
clf = ComplementNB()

# Convert X_train from sparse matrix to array
X_train_array = X_train.toarray()

# Convert array elements to strings
X_train_texts = [' '.join(map(str, row)) for row in X_train_array]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_texts)

# Convert X_test from sparse matrix to array
X_test_array = X_test.toarray()

# Convert array elements to strings
X_test_texts = [' '.join(map(str, row)) for row in X_test_array]

# Transform the test data using the fitted vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test_texts)

# Define hyperparameters to search
param_grid = {'alpha': [0.1, 0.5, 1.0]}

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model
best_clf = grid_search.best_estimator_

# Fit the best model on the training data
best_clf.fit(X_train, y_train)

# Predict on the test data
y_pred = best_clf.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy}")

import matplotlib.pyplot as plt

# Get the values of alpha and corresponding mean test scores
alphas = [0.1, 0.5, 1.0]
mean_test_scores = grid_search.cv_results_['mean_test_score']

# Plot alpha vs. accuracy
plt.figure(figsize=(8, 6))
plt.plot(alphas, mean_test_scores, marker='o')
plt.title('Alpha vs. Accuracy Score')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Accuracy')
plt.grid(True)
plt.show()
