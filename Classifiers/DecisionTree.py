# Necessary Imports:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Decision Tree Function:
def DecisionTree(dataset_type,X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    depths = range(1, 21)  
    accuracy_scores = []

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

    val_accuracy = max(accuracy_scores)
    print("For dataset type = "+dataset_type)
    print(f"Maximum obtained Accuracy: {val_accuracy}")

    plt.figure(figsize=(6, 4))
    plt.plot(depths, accuracy_scores, marker='o')
    plt.title('Decision Tree Model Complexity vs. Accuracy')
    plt.xlabel('Max Depth of Tree')
    plt.ylabel('Accuracy on Test Set')
    plt.grid(True)
    plt.show()

    return val_accuracy