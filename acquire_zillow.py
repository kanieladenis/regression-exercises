import pandas as pd



# create function 'get_connection' for repeated use to pass authentication to MySQL server
def get_connection(db_name):
    '''
   This function used the passed database name and imports host, user, password
   from the locally saved env file to authenticate with the MySQL server.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


# create function 'get_telco' to pull all records from all tables from telco_churn database..
def get_zillow():
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
    
