from sqlalchemy import create_engine

import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def load_data(database_filepath):
    """
    INPUTS:
    database_filepath: Path to disaster response database

    Outputs:
    X: messages
    Y: categories
    category_names: category column names
    """

    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql('categorized_messages', con=engine.connect())

    X = df['message'].values
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X, Y, category_names



def tokenize(text):
    """
    INPUTS:
    text: raw text

    OUTPUTS:
    clean tokenized text
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    #transform to lower case and remoce special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens



def build_model(estimator=AdaBoostClassifier()):
    """
    INPUTS:
    estimator: Estimator for the MultiOutputClassifier, default: AdaBoostClassifier()

    OUTPUTS: 
    cv: Pipeline with optimum parameters
    """

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mtc', MultiOutputClassifier(estimator))
    ])
    
    parameters = {
        'mtc__estimator__learning_rate': [1.0, 2.0],
        'mtc__estimator__n_estimators': [50, 100],
        }
    
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUTS:
    model: Classification model
    X_test: Testset of X
    Y_test: Tesset of Y
    category_names: names of categories

    OUTPUTS:
    None
    """
    Y_Predict = model.predict(X_test)

    print(classification_report(Y_test, Y_Predict, target_names=category_names))



def save_model(model, model_filepath):
    """
    INPUTS:
    model: trained model
    model_filepath: Path where to save the pickle file with the model

    OUTPUTS:
    None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()