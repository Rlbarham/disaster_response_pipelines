import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

# configure libraries
stop_words = nltk.corpus.stopwords.words("english")
stop_words.append('us')
stop_words.append('000')
stop_words.append('http')


def load_data(database_filepath):
    """
    Reads data from a Database

    Args:
    database_filepath: path to SQL database

    Returns:
    X: Features dataframe
    Y: Target dataframe
    category_names: A list of names for target labels 
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message', con=engine)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes a text input
    
    Args:  
        text: Source text to be tokenized
        
    Returns:
        tokenized_text: The source text after the transformation
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return tokenized_text


def build_model():
    """
    Build model and tune it
    
    Returns:
    Trained model
    """
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=tokenize)),
    ('mo_clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=2)))
    ])

    parameters = {
    'mo_clf__estimator__max_depth': [None, 3],
    'mo_clf__estimator__min_samples_split': [2, 4],
    'vect__max_df': (0.7, 1.0),
    }

    model = GridSearchCV(pipeline, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Display how the model performs on the test date

    Args:
    model: trained model for evaluation
    X_test: features for the test
    Y_test: target variables for the test
    category_names: category labels
    """
    y_pred = model.predict(X_test)
    
    output_df = pd.DataFrame(columns=['Category', 'Precision', 'Recall', 'F1_Score'])    
    cat_list = category_names
    tracker = 0
    for item in cat_list:
        precision, recall, f1_score, support = precision_recall_fscore_support(Y_test[item], y_pred[:,tracker], average='weighted')
        output_df.at[tracker+1, 'Category'] = item
        output_df.at[tracker+1, 'Precision'] = precision
        output_df.at[tracker+1, 'Recall'] = recall
        output_df.at[tracker+1, 'F1_Score'] = f1_score
        tracker = tracker + 1
        
    # print aggregated outputs
    print('Mean precision:', output_df['Precision'].mean())
    print('Mean recall:', output_df['Recall'].mean())
    print('Mean f1_score:', output_df['F1_Score'].mean())



def save_model(model, model_filepath):
    """
    Stores the model as a pickle file for future use   
    
    Args:
    Model: Final trained model to be stored
    Model_filepath: Filepath to save the model as
    """
    joblib.dump(model,  'model.pkl', compress=3)



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