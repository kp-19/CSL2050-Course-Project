# Necessary Imports:
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def linearSVC_performance(dataset_type, X_train, X_test, y_train, y_test):
    # Create a Linear SVM classifier
    svm = LinearSVC()

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Calculate accuracy
    accuracy_svc = accuracy_score(y_test, y_pred)
    print("Accuracy (Linear SVM) for " + dataset_type, accuracy_svc)
    return accuracy_svc

def linearKernelSVC_performance(dataset_type, X_train, X_test, y_train, y_test):
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_linear = svm_linear.predict(X_test)

    # Calculate accuracy
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print("Accuracy (Linear Kernel) for "+dataset_type, accuracy_linear)
    return accuracy_linear

def polyKernelSVC_performance(dataset_type, X_train, X_test, y_train, y_test):
    # Create an SVM classifier with polynomial kernel
    svm_poly = SVC(kernel='poly',C=5,degree=7)

    svm_poly.fit(X_train, y_train)
    y_pred_poly = svm_poly.predict(X_test)

    # Calculate accuracy
    accuracy_poly = accuracy_score(y_test, y_pred_poly)
    print("Accuracy (Polynomial Kernel) for "+dataset_type, accuracy_poly)
    return accuracy_poly

def rbfKernelSVC_performance(dataset_type, X_train, X_test, y_train, y_test):
    # Create an SVM classifier with rbf kernel
    svm_rbf = SVC(kernel='rbf',C=5,gamma='scale')

    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)

    # Calculate accuracy
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print("Accuracy (RBF kernel) for "+dataset_type, accuracy_rbf)
    return accuracy_rbf