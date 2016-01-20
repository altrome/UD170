import numpy as np
import pandas
import matplotlib.pyplot as plt

def entries_histogram(filename):
    '''
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's 
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.
    
    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()
    
    Your histograph may look similar to bar graph in the instructor notes below.
    
    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms
    
    You can see the information contained within the turnstile weather data here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    '''
    turnstile_weather_0 = pandas.read_csv(filename)
    plt.figure()
    turnstile_weather = turnstile_weather_0[turnstile_weather_0['ENTRIESn_hourly'] < 6000]
    turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 1].plot(kind='hist', bins=21, color='g', legend=True, label='Rain') # your code here to plot a historgram for hourly entries when it is raining
    turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 0].plot(kind='hist', bins=25, color='b', legend=True, label='No Rain', alpha=0.5) # your code here to plot a historgram for hourly entries when it is not raining
    plt.show()
    #return plt

entries_histogram('turnstile_data_master_with_weather.csv')
