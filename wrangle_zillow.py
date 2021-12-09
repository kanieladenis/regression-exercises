from env import user, password, host
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.feature_selection import SelectKBest, f_regression

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
                                       

def x_y(train, validate, test, target):
    '''
    this function creates the X and y version for train, validate, and test
    '''
    # create X & y version of train, where y is a series with just the target variable and X are all the features. 
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]
    return X_train, X_validate, X_test, y_train, y_validate, y_test
    

    
    
def scaler(X_train, X_validate, X_test):
    '''
    This function uses Min Max Scaler to scale X version for train, validate, test
    '''
    # Define the thing
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit the thing
    scaler.fit(X_train)

    # create X versions scaled
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert numpy array to pandas dataframe for feature Engineering
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns.to_list())
    X_validate_scaled = pd.DataFrame(X_validate_scaled, index=X_validate.index, columns=X_validate.columns.to_list())
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns.to_list())
    
    return X_train_scaled, X_validate_scaled, X_test_scaled


def select_kbest(X_train_scaled, y_train, k):
    # parameters: f_regression stats test, give me 2 features
    f_selector = SelectKBest(f_regression, k=k)

    # find the top 2 X's correlated with y
    f_selector.fit(X_train_scaled, y_train)

    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()

    return f_feature


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow())
    
    return train, validate, test
    