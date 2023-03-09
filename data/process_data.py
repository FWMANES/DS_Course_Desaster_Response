import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    Input:
    messages_filepath:  path to messages csv-file
    categories_filepath: path to categories csv-file

    Output:
    merged DataFrame comprising the messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on='id')



def clean_data(df):
    """
    Input: 
    df: merged DataFrame with messanges and categories

    Output: 
    df: cleaned DataFrame with one colum for each categorie
    """
    #df = df

    # split castegories into separate category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # replace column names 
    category_colnames = [re.sub("\-.*", "", cat) for cat in df['categories'].str.split(pat=';', expand=True).iloc[0,:]]
    categories.columns = category_colnames

    # convert categories to just numbers 0 and 1
    for index, col in enumerate(categories.columns):
        categories[col] = categories.iloc[:,index].apply(lambda x: re.sub("[^0-9]", "", x))

    categories = categories.apply(pd.to_numeric)

    # replace 2 by 1, should be a mistake during data entry
    categories.replace(2, 1, inplace=True)

    # replace categories column in df 
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.drop_duplicates(subset=['id'], inplace=True)

    return df



def save_data(df, database_filepath):
    """
    INPUT:
    df: clean DataFrame with seperate categories columns
    database_filepath: Database name

    Output:
    Database with cleaned messages and seperate categories columns
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('categorized_messages', engine, index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

