#Importing Necessary Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import wordcloud
import warnings
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
nltk.download('stopwords')
warnings.filterwarnings('ignore')

#Importing Models:
from joblib import load
dt_clf = load(filename="Trained_models\Decision_tree_tfid.joblib")
rf_clf = load(filename="Trained_models\Random_forest_tfid_100_trees.joblib")
nb_clf = load(filename="Trained_models\ComplimentNB_tfid.joblib")
svm_clf = load(filename="Trained_models\SVM_linear_kernel_tfid.joblib")
perc_clf = load(filename="Trained_models\Perceptron_tfid.joblib")
lr_clf = load(filename="Trained_models\Logistic_regression_tfid.joblib")

from nltk.stem import PorterStemmer
import re

def stem_text(txt_input):
    # Initialize Porter Stemmer
    stemmer = PorterStemmer()
    
    txt_series = pd.Series(txt_input)
    # Apply stemming to each word in the input text
    stemmed_txt = txt_series.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    
    return stemmed_txt

def preprocess_input(text, age, time_of_tweet):
    # Remove special characters, numbers, and other punctuations
    processed_text = text.replace("[^a-zA-Z#]", " ")
    
    # Convert processed text to lowercase
    processed_text = processed_text.lower()
    
    # Apply Porter stemming
    processed_text = stem_text(processed_text)
    
    # Create a DataFrame to hold the preprocessed data
    processed_data = pd.DataFrame({'processed_text': processed_text, 'age': age, 'time_of_tweet': time_of_tweet})
    
    return processed_data

from statistics import mode

def ensemble_predict(X_test, dt_clf, rf_clf, nb_clf, svm_clf, perc_clf, lr_clf):
    X_test_array = X_test.toarray()

    # Initialize lists to store predictions from each classifier
    dt_predictions = []
    rf_predictions = []
    svm_predictions = []
    lr_predictions = []
    nb_predictions = []
    perc_predictions = []

    # Getting the predictions:
    for data_point in X_test_array:
        dt_pred = dt_clf.predict(data_point.reshape(1, -1))
        dt_predictions.append(dt_pred[0])    
    rf_predictions = rf_clf.predict(X_test)
    nb_predictions = nb_clf.predict(X_test)
    perc_predictions = perc_clf.predict(X_test)
    lr_predictions = lr_clf.predict(X_test)
    svm_predictions = svm_clf.predict(X_test)
    
    # Initialize list to store final ensemble predictions
    ensemble_predictions = []

    # Combine predictions from all classifiers
    for i in range(len(X_test_array)):
        # Calculate mode label from predictions of all classifiers
        mode_label = mode([dt_predictions[i], rf_predictions[i], lr_predictions[i], nb_predictions[i], perc_predictions[i], svm_predictions[i]])
        ensemble_predictions.append(mode_label)

    return ensemble_predictions

vectorizer = load(filename="Trained_models\Vectorizer.joblib")

def predict_label(text, age, time_of_tweet):
    # Preprocess input
    processed_data = preprocess_input(text, age, time_of_tweet)
    
    # Vectorize the processed text
    vectorized_input = vectorizer.transform(processed_data['processed_text'])
    
    # Concatenate the age and time_of_tweet features with the vectorized text
    X_input = sparse.hstack([processed_data[['age', 'time_of_tweet']], vectorized_input])

    # Make predictions using the AdaBoost classifier
    predictions = ensemble_predict(X_input, dt_clf, rf_clf, nb_clf, svm_clf, perc_clf, lr_clf)
    
    return predictions

from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
@app.route('/')
def man():
    return render_template('webpage.html')


@app.route('/predict', methods=['POST'])
def home():
    user_text = request.form['a']
    age = request.form['b']
    time = request.form['c']
    pred = predict_label(user_text, int(age), int(time))[0]
    if(int(pred) == 0):
        return render_template('webpage.html', data="Negative")
    elif(int(pred) == 1):
        return render_template('webpage.html', data="Neutral")
    else:
        return render_template('webpage.html', data="Positive")
    
if __name__ == "__main__":
    app.run(debug=True)

