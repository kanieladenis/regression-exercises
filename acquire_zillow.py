import pandas as pd
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
    
