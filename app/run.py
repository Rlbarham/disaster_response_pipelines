import json
import plotly
import numpy as np
import pandas as pd
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import operator

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine

# configure libraries
stop_words = nltk.corpus.stopwords.words("english")
stop_words.append('us')
stop_words.append('000')
stop_words.append('http')

app = Flask(__name__)

def tokenize(text):
    # convert text to lower case
    text = text.lower()
    
    # remove punctuation with a regex
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # remove stop words
    filtered_tokens = [w for w in tokens if not w in stop_words]
    
    # lemmatize words
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return tokenized_text

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # 1. genre count chart
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 2. potential request count chart
    request_counts = df.groupby('request').count()['message']
    request_cats = list(request_counts.index)

    # 3. translation chart
    trans_cat_names = ['Translated', 'Not Translated']
    trans_counts = [df.original.notnull().sum(), df.original.isnull().sum()]

    # 4. text count chart
    # extract and sort words
    tokenized_holder = []                             
    for item in df['message'].values:
        tokenized_holder.extend(tokenize(item))
    words_by_count = dict(sorted(Counter(tokenized_holder).items(), key=operator.itemgetter(1), reverse=True))
    
    # filter by top 25 words available
    limit = 25
    counter = 0
    top_words = {}

    for key, value in words_by_count.items():
        if counter > limit:
            break
        else:
            top_words[key] = value
            counter += 1

    # prepare variables for chart
    words = list(top_words.keys())
    count_props = list(top_words.values())

    # create visuals
    # base chart
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },


        # translation chart
        {
            'data': [
                Pie(
                    labels=trans_cat_names,
                    values=trans_counts 
                )
            ],

            'layout': {
                'title': 'Proportion of messages translated',
            }
        },

        # word count chart
        {
        'data': [
            Bar(
                x=words,
                y=count_props
            )
        ],

            'layout': {
                'title': 'Top 25 words by frequency',
                'yaxis': {
                    'title': 'Count',
                },
                'xaxis': {
                    'title': 'Words',
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()