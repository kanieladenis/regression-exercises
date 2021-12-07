def get_residuals(df):
    
    X = df[['total_bill']]
    y = df[['tip']]
   
    # Create Linear Regression Container and fit to data
    ols_model = LinearRegression().fit(X,y)

    # Create prediction column based on feature
    df['yhat'] = ols_model.predict(y)

    # create residiaul column as differnce of prediction and actual
    df['residual'] = df.yhat - df.tip
    
    return df
    
    
    
# plot_residuals(y, yhat): creates a residual plot
def plot_residuals(df):
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=df, x=df.total_bill, y=df.residual)
    plt.show()
    
    
    
def regression_errors(df):
    
    # create residual squared
    df = df['residual^2'] = df.residual**2

    # create sum of squared residuals
    sse = df['residual^2'].sum()

    # find mean of squared residuals
    mse = sse/len(df)
    
    # find root of the mean of squared residuals
    rmse = sqrt(mse)
    
    # find explained square sum of errors
    ess = sum((df.yhat - df.tip.mean())**2)
    
    # fine total sum squared of errors
    tss = sum((df.tip - df.tip.mean())**2)
    
    return df, sse, mse, rmse, ess, tss


def baseline_mean_errors(df):
    df['baseline'] = df.tip.mean()

    df['residual_baseline'] = df.baseline - df.tip

    df['residual_baseline^2'] = df.residual_baseline ** 2

    baseline_sse = sum(df['residual_baseline^2'])

    baseline_mse = baseline_sse/len(df)

    baseline_rmse = sqrt(baseline_sse)

    return df, baseline_sse, baseline_mse, baseline_rmse

    
    
    
def better_than_baseline(rmse, baseline_rmse):
    return print(f' model is better than baseline : {rmse < baseline_rmse}')
    
    