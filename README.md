# Disaster Response Pipeline Project

## Installation
This project was built in Python 3.6. A number of additional libraries were also used - these libraries and their version numbers are listed below. 

- numpy: 1.16.2
- pandas: 0.24.2
- scikit-learn: 0.20.3
- nltk: 3.4.4
- flask: 1.0.2
- plotly: 4.1.0
- sqlalchemy: 1.3.1

## Project overview and motivation

This project features a basic web application to analyze and provide specific and summary feedback on text responses. 

The project draws on a data set from Figure Eight, containing real messages sent during disaster events. The application provides a user the capability to classify a new disaster message, and provides some high-level analysis on the data provided (e.g. how many original messages were translated).

To do this, the project draws on some underlying ETL and ML pipelines that were initially prototyped in a notebook format. 

## File descriptions
- \
	- README.md
- \app
	- run.py
	- \templates
	   - go.html
	   - master.html
- \data
	- DisasterResponse.db
	- disaster_categories.csv
	- disaster_messages.csv
	- process_data.py
- \models
	- model.pkl (too big to upload to GH)
	- train_classifier.py
- \notebooks
	- ETL Pipeline Preparation.ipynb
	- ML Pipeline Preparation.ipynb

## Instructions to run

Running the application is simple.

To run the ETL pipelines and ML pipelines, use process_data.py and train_classifier.py respectively.

To run the live application, run 'python run.py' in the relevant directory. Once the application has loaded on a local server, visit the relevant URL (by default http://0.0.0.0:3001/)

### Licensing, Authors, Acknowledgements, etc.
The project has been completed by me directly. The raw data is sourced from Figure Eight. 