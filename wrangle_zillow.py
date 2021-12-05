from env import user, password, host
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

import os



# create function 'get_connection' for repeated use to pass authentication to MySQL server
def get_connection(db_name):
    '''
   This function used the passed database name and imports host, user, password
   from the locally saved env file to authenticate with the MySQL server.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
    


# Uses get_connection function pull data from sql server
def get_new_zillow():
    '''
    This function uses the the get_connection function to pull the following columns from zillow: bedroomcnt, bathroomcnt,
    calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips.
    '''
    sql = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017 p17
    JOIN propertylandusetype bct using (propertylandusetypeid)
    WHERE propertylandusetypeid="261"
    '''
    
    url = get_connection('zillow')
    df = pd.read_sql(sql, url)
    return df


# get zillow data by reading from csv if available or pull from server if not
def get_zillow():
    file = 'zillow_data.csv'
    if os.path.isfile(file):
        df = pd.read_csv(file, index_col=0)
    else:
        df = get_new_zillow()
        df.to_csv(file)
    return df


def clean_zillow(df):
    
    # renaming columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'area',
                          'taxvaluedollarcnt':'tax_value',
                              'taxamount':'tax_amount', 
                          'yearbuilt':'year_built'})
    return df


def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def prepare_zillow(df):
    '''
    '''
    
    #rename columns
    df = clean_zillow(df)
    
    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount'])
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test    
                                       


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow())
    
    return train, validate, test
    