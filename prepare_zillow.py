import sklearn.preprocessing

def scale_zillow(train, validate, test):
    '''
    '''
    # scale tax amount and tax value by min-max scaler
    # create the object
    scaler_norm = sklearn.preprocessing.MinMaxScaler()

    # fit the object (learn the min and max value)
    scaler_norm.fit(train[['tax_value', 'tax_amount']])

    # use the object (use the min, max to do the transformation)
    train = scaler_norm.transform(train[['tax_value', 'tax_amount']])
    validate = scaler_norm.transform(validate[['tax_value', 'tax_amount']])
    test = scaler_norm.transform(test[['tax_value', 'tax_amount']])
    
    return train, validate, test