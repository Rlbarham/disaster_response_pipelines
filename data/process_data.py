import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Reads in required messages and category data to run our application

    Args:
        messages_filepath: The path to locate the messages data input
        categories_filepath (string): The path to locate the categories data input
    
    Returns:
        A dataframe object obtained by merging the input dataframes
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Cleans dataframe to prepare it for modelling
      
    Args: 
      df: a merged dataframe reflecting messages and categories data
      
    Returns:
      df: a cleaned dataframe 
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = []
    for entry in row:
        category_colnames.append(entry[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
      categories[column] = categories[column].astype(str).str[-1]
      categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save final cleaned data set to a SQLite database

    Args:
      df: Cleaned data from preceding clean_data function
      database_file_name str: Path for SQLite database
    """

    # create a sqlite database 
    engine = create_engine('sqlite:///Messages.db')

    # transfer dataframe into sqlite database
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')
    


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