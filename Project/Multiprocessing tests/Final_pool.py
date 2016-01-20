import numpy as np
import scipy.stats as sps
import pandas as pd
import statsmodels.api as sm
import itertools
from ggplot import *
from multiprocessing import Pool



def create_dataframe():
    datafile='turnstile_weather_v2.csv'
    df = pd.read_csv(datafile)
    #df['rain'] = df['rain'].apply(lambda x: True if x==1 else False)
    df['station'] = df['station'].apply(lambda x: x.replace(' ', '').replace('-', ''))
    df['conds'] = df['conds'].apply(lambda x: x.replace(' ', ''))
    #print df.ix[:, [0,2,4]]
    #print df.iloc[:, [0,2,4]]
    return df

def mann_whitney():
    df = create_dataframe()
    rain_df = df['ENTRIESn_hourly'][df['rain'] == True]
    no_rain_df = df['ENTRIESn_hourly'][df['rain'] == False]
    rain_mean = np.mean(rain_df)
    no_rain_mean = np.mean(no_rain_df)
    mannwhitneyu = sps.stats.mannwhitneyu(rain_df, no_rain_df)
    print 'Rain Mean: ', rain_mean,' No Rain Mean: ',no_rain_mean
    print 'Mann-Whithney Resuts: U=', mannwhitneyu[0],' p-value(two-tail)=',mannwhitneyu[1]*2

    
    
def plot_dhist():
    df = create_dataframe()
    plot = ggplot(df, aes(x='ENTRIESn_hourly', color='rain', fill='rain')) + geom_histogram(binwidth=1000) + \
           ggtitle('Histogram of Hourly Entries') + \
           labs('Entries', 'Freq')
    print plot

    
def linear_regression(features, values):
    
    features = sm.add_constant(features)
    model = sm.OLS(values,features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    
    return intercept, params

def compute_r_squared(data, predictions):
    den = ((data-np.mean(data))**2).sum()
    num = ((predictions-data)**2).sum()
    r_squared = 1 - num / den
    return r_squared

def predictions(df,features):	
    # Add 'stations' to features using dummy variables
    dummy_stations = pd.get_dummies(df['station'], prefix='station')
    features = features.join(dummy_stations)
    # Add 'cond' to features using dummy variables
    dummy_conds = pd.get_dummies(df['conds'], prefix='cond')
    features = features.join(dummy_conds)
    
    # Values
    values = df['ENTRIESn_hourly']
    
    # Get the numpy arrays
    features_array = features.values
    values_array = values.values
    # Perform linear regression
    intercept, params = linear_regression(features_array, values_array)
    
    predictions = intercept + np.dot(features_array, params)
    #return predictions
    print compute_r_squared(values, predictions)
    return compute_r_squared(values, predictions)


def search_max_r_squared():
    df = create_dataframe()   
    # Select Features (try different features!)
    pool = Pool(4)
    all_features = [
        'hour',
        'day_week',
        'weekday',
        'latitude',
        'longitude',
        'fog',
        'precipi',
        'pressurei',
        'rain',
        'tempi',
        'wspdi',
        'meanprecipi',
        'meanpressurei',
        'meantempi',
        'meanwspdi',
        'weather_lat',
        'weather_lon'
    ]
    r = 1
    max_r_sq = 0
    results = []
    #while r <= len(all_features):
    print 'Starting combinations'
    while r <= 4:
        list_of_combinations = list(itertools.combinations(all_features, r))
        for list_of_features in list_of_combinations:
            #print 'Adding ',list_of_features,' combination to the Queue'
            features = df[np.asarray(list_of_features)]
            results.append(pool.apply_async(predictions, args=(df,features)))
            #if actual_r_sq > max_r_sq:
                #print 'New max found: ', actual_r_sq,' with the combination ',  list_of_features
                #max_r_sq = actual_r_sq
        r += 1
    print 'Printing outputs:'
    output = [p.get() for p in results]
    print output
    
search_max_r_squared()
