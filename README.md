# DS_Course_Disaster_Response

 

## 1.	Installation
This analysis was performed with:
-	python 3.10
-	pandas 1.4.3
-	numpy 1.23.1
-	scikit-learn 1.1.2
-	SQLAlchemy 2.0.4
-	Regex 2022.10.31
-	Nltk 3.8.1
-	pickleshare 0.7.5
-	plotly 5.13.0
-	Flask 2.2.2

## 2.	Project Motivation

In this project a model pipeline is build which classifies the messages sent during disaster as one of 36 categories which were already defined. The data is provided by Appen (formally figure 8). The project comprises an ETL-Pipeline to load the datasets, merge them, prepare the data and store them into a SQL database. Afterwards a machine learning model is trained to classify new disaster messages after preprocessing the messages and saved into a pickle-file.

## 3.	File Description
- **Preparation**
    - categories.csv
    - ETL Pipeline Preparation.ipynb
    - messages.csv
    - ML Pipeline Preparation.ipynb
    - test.db
    - test.pkl
- **app**					
    - **templates**
        - go.html
        - master.html
    - run.py
- **data**
    - disaster_categories.csv
    - disaster_messages.csv
    - process_data.py
- **models**
    - train.py

- README.md

## 4.	How to interact with this project

To create the database with the disaster messages start the process_data.py script in the data folder. The necessary Inputs are the path to a csv with the messages, the path to the categories and the database name.
The machine learning model is trained with the train.py script in the model folder. Inputs are the path to the disaster messages database and the path to the pickle file where the model will be saved.
To start the web app with a visualization of the results start the run.py script in the app folder.

## 5.	Licenses, Authors

n/a
