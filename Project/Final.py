import numpy as np
import scipy.stats as sps
import pandas as pd
import statsmodels.api as sm
import itertools
import datetime
from ggplot import *
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

'''
Global vars definition
'''
#features list
all_features = [
    #'hour', # discrete var --> Dummy values[0, 4, 12, 16, 20]
    #'station', # discrete var --> Dummy 
    #'conds', #discrete var --> Dummy values['Clear', 'Fog', 'Haze', 'HeavyRain', 'LightDrizzle', 'LightRain', 'Mist', 'MostlyCloudy', 'Overcast', 'PartlyCloudy', 'Rain', 'ScatteredClouds']
    #'day_week', #discrete var --> Dummy values[0, 1, 2, 3, 4, 5, 6]
    #'weekday', #Highly correlated with 'day_week'
    #'latitude', #Highly correlated with 'station'
    #'longitude', #Highly correlated with 'station'
    #'rain', #Highly correlated with 'precipi'
    'fog',
    'precipi',
    'pressurei',
    'tempi',
    #'wspdi',
    #'meanprecipi',
    #'meanpressurei',
    #'meantempi',
    #'meanwspdi',
    'weather_lat',
    'weather_lon'
]

'''
FUNCTIONS
'''

#DataFrame Creation
#input: NONE
#output: dataframe
def create_dataframe():
    datafile='improved-dataset/turnstile_weather_v2.csv'
    df = pd.read_csv(datafile)
    #df['rain'] = df['rain'].apply(lambda x: True if x==1 else False)
    df['station'] = df['station'].apply(lambda x: x.replace(' ', '').replace('-', ''))
    df['conds'] = df['conds'].apply(lambda x: x.replace(' ', ''))
    #l = ['Clear', 'Fog', 'Haze', 'HeavyRain', 'LightDrizzle', 'LightRain', 'Mist', 'MostlyCloudy', 'Overcast', 'PartlyCloudy', 'Rain', 'ScatteredClouds']
    #conds = dict(zip(l,(1,2,3,4,5,6,7,8,9,10,11,12)))
    #df['conds'] = df['conds'].map(conds)
    #print df
    #print df.ix[:, [0,2,4]]
    #print df.iloc[:, [0,2,4]]
    #df = df[df.weekday == 1]
    return df

#Dummy var addition
#input: df=dataframe, features=features array
#output: features array with dummy variables
def add_dummy_vars(df, features):
    # Add 'stations' to features using dummy variables
    dummy_stations = pd.get_dummies(df['station'], prefix='station')
    features = features.join(dummy_stations)
    # Add 'conds' to features using dummy variables
    dummy_conds = pd.get_dummies(df['conds'], prefix='cond')
    features = features.join(dummy_conds)
    # Add 'day_week' to features using dummy variables
    dummy_day = pd.get_dummies(df['day_week'], prefix='day_')
    features = features.join(dummy_day)
    # Add 'hour' to features using dummy variables
    dummy_hour = pd.get_dummies(df['hour'], prefix='hour_')
    features = features.join(dummy_hour)   
    return features

#Mann-Whitney Calculation
#input: NONE
#output: NONE (prints out the mean values of the samples and the U and p value)
def mann_whitney():
    df = create_dataframe()
    rain_df = df['ENTRIESn_hourly'][df['rain'] == True]
    no_rain_df = df['ENTRIESn_hourly'][df['rain'] == False]
    rain_mean = np.mean(rain_df)
    no_rain_mean = np.mean(no_rain_df)
    mannwhitneyu = sps.stats.mannwhitneyu(rain_df, no_rain_df)
    print 'Rain Mean: ', rain_mean,' No Rain Mean: ',no_rain_mean
    print 'Mann-Whithney Resuts: U=', mannwhitneyu[0],' p-value(two-tail)=',mannwhitneyu[1]*2   
    
#OLS Linear Regresson
#input: features=features array, values=values array
#output: intercept & params of the OLSmodel
def linear_regression_OLS(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values,features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    return intercept, params

#R-Square calculation
#input: data & predictions
#output: R-square (float)
def compute_r_squared(data, predictions):
    den = ((data-np.mean(data))**2).sum()
    num = ((predictions-data)**2).sum()
    r_squared = 1 - num / den
    return r_squared

#Predictions calculation
#input: NONE
#output: NONE (prints out the R-square value for a given dataset & features)
def predictions():
    df = create_dataframe()   
    features = df[all_features]

    # Add Dummy-vars
    features = add_dummy_vars(df, features)
    
    # Values
    values = df['ENTRIESn_hourly']
    
    # Get the numpy arrays
    features_array = features.values
    values_array = values.values
    # Perform linear regression
    intercept, params = linear_regression_OLS(features_array, values_array)
    #print intercept, params
    predictions = intercept + np.dot(features_array, params)
    return predictions
    print compute_r_squared(values, predictions)

'''
iterate to find the max r_squared
'''
#Predictions calculation
#input: NONE
#output:R-square value for a given dataset & features
def predictions_iter(df,features):

    # Add Dummy-vars
    features = add_dummy_vars(df, features)

    # Values
    values = df['ENTRIESn_hourly']
    
    # Get the numpy arrays
    features_array = features.values
    values_array = values.values
    # Perform linear regression
    intercept, params = linear_regression_OLS(features_array, values_array)
    
    predictions = intercept + np.dot(features_array, params)
    #return predictions
    return compute_r_squared(values, predictions)

#iterate prediction calculation with different combinations of r features
#input: r= number of features in the combination
#output:Dataframe with the R-square and the features dropped from the model
#prints out the combination used and the R-square 
def search_max_r_squared(r):
    df = create_dataframe()   
    max_r_sq = 0
    results = []
    list_of_combinations = list(itertools.combinations(all_features, r))
    for list_of_features in list_of_combinations:
        features = df[np.asarray(list_of_features)]
        actual_r_sq = predictions_iter(df,features)
        results.append((actual_r_sq,list(set(all_features) - set(list_of_features))))
        print list_of_features,',',actual_r_sq
        if actual_r_sq > max_r_sq:
            print 'New max found: ', actual_r_sq
            max_r_sq = actual_r_sq
    output = pd.DataFrame(results).rename(columns={0:'r_square', 1:'dropped'})
    output['dropped'] = output['dropped'].apply(lambda x: ', '.join(x))
    output = output.reset_index()
    return output

#iterate prediction calculation with different combinations with any number of features
#input: r=number of features in the combination OPTIONAL
#output:R-square value for a given dataset & features
def search_max_r_squared_iter(r=1):
    df = create_dataframe()   
    max_r_sq = 0
    while r <= len(all_features):
        list_of_combinations = list(itertools.combinations(all_features, r))
        for list_of_features in list_of_combinations:
            print 'Trying ',list_of_features,' combination'
            features = df[np.asarray(list_of_features)]
            actual_r_sq = predictions_iter(df,features)
            if actual_r_sq > max_r_sq:
                print 'New max found: ', actual_r_sq,' with the combination ',  list_of_features
                max_r_sq = actual_r_sq
        r += 1

'''
linear regression SDG
'''

#SGD Linear Regresson
#input: features=features array, values=values array
#output: intercept & params of the SGD model
def linear_regression_SGD(features, values):
    
    model = SGDRegressor(n_iter=15)
    model.fit(features, values)
    intercept = model.predict(np.zeros(np.shape(features)[1]))
    params = model.predict(np.identity(np.shape(features)[1]))-intercept
    #coeficients = model.decision_function(features)
    #intercept = coeficients[0]
    #params = coeficients[1:]
    
    return intercept, params

def normalize_features(features):
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

#Predictions calculation SGD
#input: NONE
#output: NONE (prints out the R-square value for a given dataset & features)
def predictions_SGD():
    df = create_dataframe()   
    features = df[all_features]
    
    # Add Dummy-vars
    features = add_dummy_vars(df, features)
    
    # Values
    values = df['ENTRIESn_hourly']

    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    means, std_devs, normalized_features_array = normalize_features(features_array)
    
    # Perform gradient descent
    norm_intercept, norm_params = linear_regression_SGD(normalized_features_array, values_array)
    
    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)
    
    predictions = intercept + np.dot(features_array, params)
    #return predictions
    print compute_r_squared(values, predictions)


'''
iterate to find the max r_squared
'''

#Predictions calculation SGD
#input: NONE
#output:R-square value for a given dataset & features
def predictions_SGD_iter(df,features):

    # Add Dummy-vars
    features = add_dummy_vars(df, features)
    
    # Values
    values = df['ENTRIESn_hourly']
    
    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    means, std_devs, normalized_features_array = normalize_features(features_array)
    
    # Perform gradient descent
    norm_intercept, norm_params = linear_regression_SGD(normalized_features_array, values_array)
    
    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)
    
    predictions = intercept + np.dot(features_array, params)
    #return predictions
    return compute_r_squared(values, predictions)

#iterate prediction calculation with different combinations of r features
#input: r= number of features in the combination
#output:NONE
#prints out the combination used and the R-square 
def search_max_r_squared_SGD(r):
    df = create_dataframe()   
    max_r_sq = 0
    list_of_combinations = list(itertools.combinations(all_features, r))
    for list_of_features in list_of_combinations:
        print 'Trying ',list_of_features,' combination'
        features = df[np.asarray(list_of_features)]
        actual_r_sq = predictions_SGD_iter(df,features)
        if actual_r_sq > max_r_sq:
            print 'New max found: ', actual_r_sq,' with the combination ',  list_of_features
            max_r_sq = actual_r_sq

#iterate prediction calculation with different combinations with any number of features
#input: r=number of features in the combination OPTIONAL
#output:R-square value for a given dataset & features
def search_max_r_squared_SGD_iter(r=1):
    df = create_dataframe()   
    max_r_sq = 0
    while r <= len(all_features):
        list_of_combinations = list(itertools.combinations(all_features, r))
        for list_of_features in list_of_combinations:
            print 'Trying ',list_of_features,' combination'
            features = df[np.asarray(list_of_features)]
            actual_r_sq = predictions_SGD_iter(df,features)
            if actual_r_sq > max_r_sq:
                print 'New max found: ', actual_r_sq,' with the combination ',  list_of_features
                max_r_sq = actual_r_sq
        r += 1

'''
Tests functions
'''

#testing model own functions vs custom funtions
def linear_regression_SGD_test():

    df = create_dataframe()   
    features = df[all_features]
    
    # Add Dummy-vars
    features = add_dummy_vars(df, features)
    
    # Values
    values = df['ENTRIESn_hourly']

    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    means, std_devs, normalized_features_array = normalize_features(features_array)
    
    model = SGDRegressor(n_iter=50)
    model.fit(normalized_features_array, values_array)

    print model.decision_function(normalized_features_array), model.score(normalized_features_array, values_array)

    #print model.get_params()
    
    norm_intercept = model.predict(np.zeros(np.shape(features)[1]))
    norm_params = model.predict(np.identity(np.shape(features)[1]))-norm_intercept
    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)   
    predictions = intercept + np.dot(features_array, params)
    print predictions ,compute_r_squared(values, predictions)


'''
ploting functions
'''

def plot_dhist():
    df = create_dataframe()
    plot = ggplot(df, aes(x='ENTRIESn_hourly', color='rain', fill='rain')) + geom_histogram(binwidth=250) + \
           ggtitle('Histogram of Hourly Entries') + \
           xlim(low=0, high=15000) + \
           labs('Entries', 'Freq')
    print plot

def plot_residuals_hist():
    df = create_dataframe()
    pred = predictions()
    residuals = df['ENTRIESn_hourly'] - pred
    #plt.figure()
    #residuals.plot(kind='hist', bins=250, color='g', legend=True, label='Residuals')
    #plt.show()
    res_df = pd.DataFrame(residuals).rename(columns={'ENTRIESn_hourly': 'Residuals'})
    #print res_df
    plot = ggplot(aes(x='Residuals'), data=res_df) + geom_histogram(binwidth=500) + \
           ggtitle('Residual Histogram') + \
           xlim(low=-15000, high=15000) + \
           labs('Residuals', 'Degree')
    print plot

def dnorm(x, mean, var):
    return sps.norm(mean,var).pdf(x)

def plot_residuals_bar():
    df = create_dataframe()
    pred = predictions()
    residuals = df['ENTRIESn_hourly'] - pred
    res_df = pd.DataFrame(residuals).rename(columns={'ENTRIESn_hourly': 'Residuals'}).reset_index()
    plot = ggplot(aes(y='Residuals', x='index'), data=res_df) + geom_bar(stat='bar', color='black') + \
           ggtitle('Residual per data point') + \
           labs('Residual', 'Point') + \
           stat_smooth(color='red')
    print plot

def plot_residuals_point():
    df = create_dataframe()
    pred = predictions()
    residuals = df['ENTRIESn_hourly'] - pred
    res_df = pd.DataFrame(residuals).rename(columns={'ENTRIESn_hourly': 'Residuals'}).reset_index()
    plot = ggplot(aes(y='Residuals', x='index'), data=res_df) + geom_point() + \
           ggtitle('Residual per data point') + \
           labs('Residual', 'Point') + \
           stat_smooth(color='red')
    print plot
    
def plot_residiuals_prob():
    pred = predictions()
    res = sps.probplot(pred, plot=plt) 
    plt.show()

def plot_entries_by_hour_station():
    df = create_dataframe()
    df_by_UNIT_sum = df.groupby(['hour', 'station']).aggregate(np.sum).reset_index()
    plot = ggplot(df_by_UNIT_sum, aes(x='hour', y='ENTRIESn_hourly', color='station')) + geom_point()  + ggtitle('Entries per Hour & station')
    print plot

def plot_entries_by_hour_day():
    df = create_dataframe()
    days_list = ['0 - Monday', '1 - Tuesday', '2 - Wednesday', '3 - Thursday', '4 - Friday', '5 - Saturday', '6 - Sunday']
    days = dict(zip((0,1,2,3,4,5,6),days_list))
    df['day_week'] = df['day_week'].map(days)
    df_by_Day_Hour = df.groupby(['hour', 'day_week', 'weekday']).aggregate(np.sum).reset_index()
    plot = ggplot(aes(x='hour', y='ENTRIESn_hourly', color='day_week', order='day_week'), data=df_by_Day_Hour) + geom_point() + \
           ggtitle('Ridership by time-of-day / day-of-week') + \
           labs('Hour','Total Entries') + \
           scale_y_continuous(limits = (0, 000)) + \
           scale_x_continuous(breaks=(0, 4, 8, 12, 16, 20), limits = (0, 24)) + \
           geom_line() 

    print plot

def plot_avg_entries_by_day_rain():
    df = create_dataframe()
    df_by_rain_Hour = df.groupby(['hour', 'rain']).aggregate(np.average).reset_index()
    print df_by_rain_Hour
    plot = ggplot(aes(x='hour', y='ENTRIESn_hourly', color='rain', fill='rain'), data=df_by_rain_Hour) + geom_point() + geom_line() + \
           ggtitle('Ridership by time-of-day / rain') + \
           labs('Hour','Average Entries') + \
           scale_y_continuous(limits = (0, 4000)) + \
           scale_x_continuous(breaks=(0, 4, 8, 12, 16, 20), limits = (0, 24))

    print plot

def plot_avg_entries_by_rain():
    df = create_dataframe()
    df_by_rain = df.groupby(['rain']).aggregate(np.average).reset_index()
    plot = ggplot(aes(x='rain', y='ENTRIESn_hourly', color='rain', fill='rain'), data=df_by_rain) + geom_bar(stat='bar') + \
           ggtitle('Ridership by time-of-day / rain') + \
           labs('rain','Average Entries') + \
           scale_y_continuous(limits = (0, 2500)) + \
           scale_x_continuous(breaks=(0, 1), limits = (-1, 2), labels=('Non Rainy', 'Rainy'))

    print plot

def plot_r_square(r):
    df = search_max_r_squared(r)
    plot = ggplot(aes(x='index', y='r_square'), df) + \
           geom_point() + geom_line() + \
           ggtitle('R Square values dropping features') + \
           labs('Dropped Feature','R Square') + \
           scale_x_continuous(breaks=range(len(df['dropped'])), labels=df['dropped']) + \
           scale_y_continuous(limits = (df['r_square'].min()*0.999, df['r_square'].max()*1.001)) + \
           theme(axis_text_x  = element_text(angle = 90, hjust = 1))
    
    print plot
