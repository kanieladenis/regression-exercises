import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# create function 'get_connection' for repeated use to pass authentication to MySQL server
def get_connection(db_name):
    '''
   This function used the passed database name and imports host, user, password
   from the locally saved env file to authenticate with the MySQL server.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


# create function 'get_telco' to pull all records from all tables from telco_churn database..
def get_telco():
    '''
    This function uses the the get_connection function to pull the telco data from the MySQL server. 
    '''
    sql = '''
    SELECT * FROM customers c
    JOIN contract_types ct ON ct.contract_type_id=c.contract_type_id
    JOIN internet_service_types ist ON ist.internet_service_type_id=c.internet_service_type_id
    JOIN payment_types pt ON pt.payment_type_id=c.payment_type_id
    '''
    url = get_connection('telco_churn')
    df = pd.read_sql(sql, url)
    return df

# Create Function to Clean Data
def clean_telco(df):
    '''
    This function cleans telco data by dropping duplicated columns, filling in blanks for 
    total charges, converts total_charge to float, creates dummy variables, and clean columns names."
    '''
    
    # dropping ducplicate columns
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    
    #replace blanks in total_charges with 0
    df['total_charges'] = df.total_charges.replace(' ', '0')
    
    # Convert total_charges to float
    df.total_charges = df.total_charges.astype(float)
    
    # Create new column for auto vs manual payment type
    df['payment'] = np.where(df.payment_type.str.contains('auto'), 'auto', 'manual')
    
    # Replace senior citizen column values with yes and no to build dummy variables
    df['senior_citizen'] = np.where(df.senior_citizen==1, 'yes', 'no')
    
    # Set Customer ID Column as Index to save for csv deliverable
    df.set_index('customer_id', inplace=True)
    
    # Id columns for dummy variables
    dummies_cols =[
        'senior_citizen',
        'gender',
        'partner',
        'dependents',
        'phone_service',
        'multiple_lines',
        'online_security',
        'online_backup',
        'device_protection',
        'tech_support',
        'streaming_tv',
        'streaming_movies',
        'paperless_billing',
        'churn',
        'contract_type',
        'internet_service_type',
        'payment']
    
    # Create dummy variables for id'd columns
    dummies_df = pd.get_dummies(df[dummies_cols], drop_first=False)
    
    # Concat dummy vairabls to df
    df = pd.concat([df, dummies_df], axis=1)
    
    # Clean column names
    df.columns = [col.lower().replace(' ','_').replace('-','_') for col in df]
    
    return df



# Create function to split data
def split_telco(df):
    '''
    This function splits the telco data into the train, validate, and test samples at 
    portions: train= 56%, validate= 24%, test = 20%
    '''
    
    # Split data to create test sample
    train_validate, test = train_test_split(df, 
                                             test_size=.2, 
                                             random_state=123, 
                                             stratify=df.churn)
    
    # Split data to create train and validate samples
    train, validate = train_test_split(train_validate,
                                      test_size=.3,
                                      random_state=123,
                                      stratify=train_validate.churn)
    
    return train, validate, test



# create funciton to prep df for modleing
def prep_telco(train, validate, test):
    '''
    This function prepares data for model ingest
    '''
    # drop uneeded columsn
    drop_cols = [
 'gender',
        'senior_citizen',
'partner',
 'dependents',
'phone_service',
 'multiple_lines',
 'online_security',
 'online_backup',
 'device_protection',
 'tech_support',
 'streaming_tv',
 'streaming_movies',
 'paperless_billing',
'churn',
 'contract_type',
 'internet_service_type',
 'payment_type',
 'gender_female',
 'gender_male',
'phone_service_no',
 'phone_service_yes',
 'multiple_lines_no',
 'multiple_lines_no_phone_service',
 'multiple_lines_yes',
 'online_backup_no',
 'online_backup_no_internet_service',
 'online_backup_yes',
 'device_protection_no',
 'device_protection_no_internet_service',
 'device_protection_yes',
'streaming_tv_no',
 'streaming_tv_no_internet_service',
 'streaming_tv_yes',
 'streaming_movies_no',
 'streaming_movies_no_internet_service',
 'streaming_movies_yes',
'monthly_charges',
 'total_charges',
'partner_yes',
 'dependents_yes',
'online_security_no_internet_service',
'tech_support_no_internet_service',
'paperless_billing_no',
'churn_no',
'contract_type_two_year',
'internet_service_type_none',
        'payment',
 'payment_auto',
    'senior_citizen_yes']
    
    # Drop columns from train
    train = train.drop(columns=drop_cols)
    
    # Drop columns from validate
    validate = validate.drop(columns=drop_cols)
    
    # Drop columns from test
    test = test.drop(columns=drop_cols)
    
    return train, validate, test

# Create X and y version
def xy_version(train,validate,test):
    '''
    This function creats the x and y version of train, validate, and test data samples
    '''
    
    # Create x and y versions of train data
    X_train = train.drop(columns=['churn_yes'])
    y_train = train.churn_yes
    
    # Create x and y versions of validate data
    X_validate = validate.drop(columns=['churn_yes'])
    y_validate = validate.churn_yes

    # Create x and y versions of test data
    X_test = test.drop(columns=['churn_yes'])
    y_test = test.churn_yes
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


