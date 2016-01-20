from pandas import *
import numpy
import datetime
from ggplot import *

def plot_weather_data(filename):
    '''
    You are passed in a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.  
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.  

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station (UNIT)
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/
     
    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
     
    To see all the columns and data points included in the turnstile_weather 
    dataframe. 
     
    However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    '''
    turnstile_weather = pandas.read_csv(filename)
    #print turnstile_weather

    #************************************
    ##### Total Entries by Unit (count)
    #tw_by_unit = pandas.DataFrame({'Entries': turnstile_weather.groupby(['UNIT']).size()}).reset_index()
    #print tw_by_unit
    #----> Plot total Entries By Unit 
    #plot = ggplot(aes(x='UNIT', y='Entries'), data=tw_by_unit) + geom_bar(stat='bar') + ggtitle('Total Entries by Unit') + ylab('Entries') 

    #************************************
    ##### Total Entries by Unit (> 1000 entries)
    #tw_by_unit_gt1000 = tw_by_unit[tw_by_unit['count'] > 1000].reset_index()
    #print tw_by_unit_gt1000
    #----> Plot total Entries by Unit (> 1000 entries)
    #plot = ggplot(aes(x='UNIT', y='Entries'), data=tw_by_unit_gt1000) + geom_bar(stat='bar') + ggtitle('Total Entries by Unit') + ylab('Entries') 

    #************************************
    ##### Total Entries by Hour
    #tw_by_Hour = pandas.DataFrame({'Entries': turnstile_weather.groupby(['Hour']).size()}).reset_index()
    #print tw_by_Hour
    #---> Plot Total Entries by Hour
    #plot = ggplot(aes(x='Hour', y='Entries'), data=tw_by_Hour) + geom_bar(stat='bar') + ggtitle('Total Entries by Hour') + ylab('Entries')
    #plot = ggplot(aes(x='Hour', y='ENTRIESn_hourly'), data=turnstile_weather) + geom_bar() + ggtitle('Total Entries by Hour') + ylab('Entries') 

    #************************************
    ##### Totals by Day
    '''
    tw_by_Day = turnstile_weather.groupby(['DATEn', 'rain', 'fog', 'thunder',
                                           'maxpressurei', 'maxdewpti', 'mindewpti',
                                           'minpressurei', 'meandewpti', 'meanpressurei',
                                           'meanwindspdi', 'mintempi', 'meantempi', 'maxtempi', 'precipi']).aggregate(numpy.sum).reset_index()
    print tw_by_Day
    '''
    ##### Totals by Day & Hour
    '''
    tw_by_Day_Hour = turnstile_weather.groupby(['DATEn', 'Hour', 'rain', 'fog', 'thunder']).aggregate(numpy.sum).reset_index()
    #print tw_by_Day_Hour
    plot = ggplot(aes(x='DATEn', y='ENTRIESn_hourly'), data=tw_by_Day_Hour) + geom_bar(stat='bar') + \
    ggtitle('Total Entries by Day&Hour') + \
    ylab('Entries') + \
    theme(axis_text_x  = element_text(angle = 90, hjust = 1))
    '''
    ##### Totals by Day of week
    '''
    turnstile_weather.is_copy = False
    turnstile_weather['DATEn'] = turnstile_weather['DATEn'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    turnstile_weather['day'] = turnstile_weather['DATEn'].apply(lambda x: x.strftime("%w"))
    tw_by_Day_week = turnstile_weather.groupby(['day']).aggregate(numpy.sum).reset_index()
    tw_by_Day_week = pandas.melt(tw_by_Day_week, id_vars=['day'], value_vars=['ENTRIESn_hourly', 'EXITSn_hourly'])
    #print tw_by_Day_week
    plot = ggplot(tw_by_Day_week, aes(x='day', y='value', color='variable', fill='variable')) + geom_bar(stat='bar')  + ggtitle('Entries/Exists per Day (0: Sunday - 6: Saturday)')
    '''

    '''
    turnstile_weather['DATEn_obj'] = turnstile_weather['DATEn'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    turnstile_weather['day'] = turnstile_weather['DATEn_obj'].apply(lambda x: x.strftime("%A"))
    tw_by_Day_Hour = turnstile_weather.groupby(['fog']).aggregate(numpy.sum).reset_index()
    print tw_by_Day_Hour

    #plot = ggplot(aes(x='DATEn', y='ENTRIESn_hourly'), data=tw_by_Day_Hour) + geom_line() + facet_wrap('day') + stat_smooth(colour="red")
    '''
    '''
    turnstile_weather.is_copy = False
    turnstile_weather['DATEn'] = turnstile_weather['DATEn'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    turnstile_weather['day'] = turnstile_weather['DATEn'].apply(lambda x: x.strftime("%A"))
    turnstile_weather['UNITn'] = turnstile_weather['UNIT'].apply(lambda x: pandas.Series(x.replace('R','')))
    turnstile_weather['UNITn'] = turnstile_weather['UNITn'].astype(int)
    tw_by_UNIT_sum = turnstile_weather.groupby(['UNITn', 'day']).aggregate(numpy.sum).reset_index()
    #print tw_by_UNIT_sum

    plot = ggplot(aes(x='UNITn', y='ENTRIESn_hourly', color='day', fill='day'), data=tw_by_UNIT_sum) + \
    geom_bar(stat='bar') + \
    ggtitle('Total Entries by Unit') + \
    ylab('Entries') + \
    xlab('UNIT') + \
    theme(axis_text_x  = element_text(angle = 90, hjust = 1)) + \
    scale_x_continuous(breaks = c(100,200,300,400))
    '''
    '''
    tw_by_UNIT_sum = turnstile_weather.groupby(['Hour', 'UNIT']).aggregate(numpy.sum).reset_index()
    tw_by_UNIT_Hour = pandas.melt(tw_by_UNIT_sum, id_vars=['UNIT', 'Hour'], value_vars=['ENTRIESn_hourly', 'EXITSn_hourly'])
    #print tw_by_UNIT_Hour
    plot = ggplot(tw_by_UNIT_Hour, aes(x='Hour', y='value', color='variable')) + geom_point() + ggtitle('Entries/Exists per Hour')
    '''

    tw_by_UNIT_sum = turnstile_weather.groupby(['Hour', 'UNIT']).aggregate(numpy.sum).reset_index()
    plot = ggplot(tw_by_UNIT_sum, aes(x='Hour', y='ENTRIESn_hourly', color='UNIT')) + geom_point()  + ggtitle('Entries per Hour & UNIT')
    print plot

    #return plot

plot_weather_data('turnstile_data_master_with_weather.csv')
