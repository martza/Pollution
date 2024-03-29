import argparse
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


from descriptive_analytics import get_coordinates_data, find_cluster, get_data_for_prediction, calibration

eea_url = 'https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw'
years_list = [str(x) for x in list(range(2013, 2021))]
# main pollutants to be used for classification
main_pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2']
# maximum pollution thresholds for each caregory in microgram/m3
# taken from https://www.eea.europa.eu/themes/air/air-quality-index
pollution_thresholds = pd.DataFrame(data={
    'pollutant': ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2'],
    'Good': [10, 20, 40, 50, 100],
    'Fair': [20, 40, 90, 100, 200],
    'Moderate': [25, 50, 120, 130, 350],
    'Poor': [50, 100, 230, 240, 500],
    'Very Poor': [75, 150, 340, 380, 750],
    'Extremely Poor': [800, 1200, 1000, 800, 1250],
})
pollution_thresholds.set_index('pollutant')
# list of variables for machine learning
ml_variables = ['Concentration', 'year', 'month', 'day', 'hour', 'weekday',
                'Longitude', 'Latitude', 'Altitude']

coordinates = pd.read_csv(
    'https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv',
    sep='\t', usecols=['AirQualityStationEoICode', 'Longitude', 'Latitude',
    'Altitude', 'AirQualityStationType', 'AirQualityStationArea']).drop_duplicates()          # list of sampling points with local coordinates


def choose_from_list(text, list):
    '''
    A FUNCTION that provides the list of options to the user and requires from
    the user to choose one of the options available.

    INPUT :
        * text : A message that requests action from the user
        * list : The list of options

    OUTPUT : value : The validated input from the user

    '''
    valid = False                                                               # Flags if the choice is valid
    while (not valid):
        choice = input(text + ', '.join(list) + '\n')
        if choice.casefold() in map(str.casefold, list):                       # Case insensitive search
            valid = True
        elif choice.casefold() == "":
            valid = True
        else:
            print('Invalid input. Choose an option from the list.')

    return choice


def get_data(countrycode, cityname, pollutantcode, year_from, year_to):
    '''
    A FUNCTION that extracts the dataset from the EEA according to the
    parameters specified by the user.

    INPUT :
        * countrycode : A string with the country-code
        * cityname : The name of the city (string)
        * pollutantcode : The numeric code corresponding to the pollutant
        chosen by the user (numeric)
        * year_from : The starting year of the dataset
        * year_to : The ending year of the dataset

    OUTPUT :
        A dataset with pollution measurements at different datetimes

    '''
    query_params = {"CountryCode": countrycode, "CityName": cityname,
                    "Pollutant": pollutantcode, "Year_from": year_from,
                    "Year_to": year_to, "Station": "", "Samplingpoint": "",
                    "Source": "E2a", "Output": "HTML", "UpdateDate": "",
                    "TimeCoverage": "Year"}
    response = requests.get(eea_url, params=query_params)

    # Checking the response of the request to the url
    print('Getting data from:\n' + response.url)
    print('The headers of the response are:\n' + ','.join(response.headers))
    print('The response is:\n' + response.text)

    # Use BeautifulSoup(kwargs) to parse urls from the response into a list
    soup = BeautifulSoup(response.content, "lxml")

    # Construct a list of urls only from the 'href' field
    urls = []
    for link in soup.find_all("a"):
        urls.append(link.get("href"))

    # Read from all urls in the list and parse to a dataframe
    dfs = (pd.read_csv(f, parse_dates=['DatetimeBegin', 'DatetimeEnd'],
                       infer_datetime_format=True) for f in urls)
    data = pd.concat(dfs, ignore_index=True)

    # Checking the characteristics of the dataset
    print('The dataset has the following column names:\n'+','.join(data.columns))
    print('The corresponding data types are:')
    print(data.dtypes)

    return data


def clean_data(data):
    '''
    A FUNCTION that cleans the dataset from NA and invalid points.

    INPUT :
        * data : The dataset
    STEPS :
        * Substitute NA in concentration with zero
        * Visualise and remove invalid points
        * Visualise unverified points
    OUTPUT :
        * A dataset with valid points
        * A visualisation of validity and verification points

    '''

    # Check for NAs
    print('There are ',sum(data.Concentration.isna()),'missing values in Concentration.')
    print(f'There are {sum(data.Validity.isna()):.0f} points missing validation data.')
    print(f'There are {sum(data.Verification.isna()):.0f} points lacking verification data.')

    # Drop NAs
    print('We remove points that are missing concentration, validity or verification data...')
    data.dropna(axis = 0, subset = ['Concentration', 'Validity', 'Verification'],
                inplace = True)

    # Visualise verified and valid points
    print('Have a look at data/check.png for invalid and unverified points')
    validity_counts = data[['Concentration', 'Validity']
                           ].groupby(['Validity']).count()
    validity_counts = validity_counts.rename(
        columns={'Concentration': 'Counts'})

    verification_counts = data[['Concentration', 'Verification']].groupby(
        ['Verification']).count()
    verification_counts = verification_counts.rename(
        columns={'Concentration': 'Counts'})

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    ax1.bar(verification_counts.index, 'Counts', data=verification_counts,
            label='Verification')
    ax1.set(xlabel='Verification', ylabel='Counts')
    ax1.legend()

    ax2.bar(validity_counts.index, 'Counts', data=validity_counts,
            label='Validity')
    ax2.set(xlabel='Validity', ylabel='Counts')
    ax2.legend()

    fig.savefig('output\check.png')

    # Remove invalid data
    print('There are', sum(data.Validity == -1),' invalid points')
    print('Removing invalid points...')
    data = data[data.Validity == 1]

    # Check for unverified points
    provisional = sum(data.Verification == 3)
    preliminary = sum(data.Verification == 2)
    verified = sum(data.Verification == 1)
    total = data.shape[0]
    print(provisional/total,' of the data are provisional.')
    print(preliminary/total,' of the data are preliminary.')
    print(verified/total,' of the data are verified.')

    # Getting rid of negative concentration values.
    stations_to_calibrate2 = data.loc[(data.Concentration < 0) & (data.Verification == 2),'AirQualityStation'].unique()
    data = calibration(data, stations_to_calibrate2)
    stations_to_calibrate3 = data.loc[(data.Concentration < 0) & (data.Verification == 3),'AirQualityStation'].unique()
    data = calibration(data, stations_to_calibrate3)

    negatives = sum(data.Concentration < 0)
    print('There are ', negatives, ' negative Concentation points in the dataset.')
    print('The maximum Concentration is ', data.Concentration.max())
    print('The minimum Concentration is ', data.Concentration.min())

    # Order data by datetime
    data = data.sort_values(by='DatetimeEnd')

    return data


def link_data(data, coordinates):
    '''
    A FUNCTION that enriches the dataset with the following information:
        * Year, month, day, hour, weekday
        * Season
        * Local coordinates, Ai Quality Station Type, Air Quality Station Area

    INPUT : the dataset

    OUTPUT : The dataset with more information
    '''

    # Add to the dataset year, day, month, hour, weekday columns
    data['year'] = pd.DatetimeIndex(data['DatetimeEnd']).year
    data['month'] = pd.DatetimeIndex(data['DatetimeEnd']).month
    data['day'] = pd.DatetimeIndex(data['DatetimeEnd']).day
    data['hour'] = pd.DatetimeIndex(data['DatetimeEnd']).hour
    data['weekday'] = data['DatetimeEnd'].dt.dayofweek

    # Definition of a new categorical variable for the seasons
    data['season'] = ''
    data.loc[data['month'].isin([12, 1, 2]), 'season'] = 'winter'
    data.loc[data['month'].isin([3, 4, 5]), 'season'] = 'spring'
    data.loc[data['month'].isin([6, 7, 8]), 'season'] = 'summer'
    data.loc[data['month'].isin([9, 10, 11]), 'season'] = 'autumn'

    # Merge coordinates with pollution concentration
    final_data = pd.merge(data, coordinates, on='AirQualityStationEoICode')

    return final_data


def EDA_pollution(data):
    '''
    A FUNCTION that explores the affect of different parameters (altitude, year,
    month, hour, weekday, season) on the
    concentration of the pollutant.

    INPUT : the dataset
    OUTPUT : plots

    '''

    # Grouping by altitude
    data_altitude = data[['Concentration', 'Altitude']].groupby(['Altitude'],
        as_index=False).aggregate([np.mean, np.median]).reset_index()
    # Grouping by altitude
    data_stationType = data[['Concentration', 'AirQualityStationType']].groupby(['AirQualityStationType'],
        as_index=False).aggregate([np.mean, np.median]).reset_index()
    # Grouping by altitude
    data_area = data[['Concentration', 'AirQualityStationArea']].groupby(['AirQualityStationArea'],
        as_index=False).aggregate([np.mean, np.median]).reset_index()

    # Checking for duplicates
    duplicates = sum(data.drop('Concentration', axis = 1).duplicated())
    print(f'Found {duplicates} duplicates.')

    plot_timeseries(data)

    # Grouping by year, month, hour, weekday, season
    yearly_data = data[['Concentration', 'year']].groupby(['year']).mean()

    monthly_data = data[['Concentration', 'year', 'month']].groupby(
        ['year', 'month']).mean()

    hourly_data = data[['DatetimeEnd', 'Concentration', 'hour']].groupby(
        ['hour'], as_index=False).aggregate(
        [np.mean, np.median]).reset_index()

    weekday_data = data[['DatetimeEnd', 'Concentration', 'weekday']].groupby(
        ['weekday'], as_index=False).aggregate(
        [np.mean, np.median]).reset_index()

    seasonal_data = data[['DatetimeEnd', 'Concentration', 'season']].groupby(
        ['season'], as_index=False).aggregate(
        [np.mean, np.median]).reset_index()

    # Plotting
    fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(10, 20))

    yearly_data['Concentration'].plot.bar(ax=ax1)
    ax1.legend(['mean'])
    ax1.set(xlabel='year', ylabel='Concentration')

    monthly_data['Concentration'].plot.bar(ax=ax2)
    ax2.legend(['mean'])
    ax2.set(xlabel='month', ylabel='Concentration')

    hourly_data.plot.bar(x='hour',
                         y=[('Concentration', 'mean'),
                             ('Concentration', 'median')],
                         rot=0, ax=ax3)
    ax3.legend(["mean", "median"])
    ax3.set(xlabel='hour of the day', ylabel="Concentration")

    weekday_data.plot.bar(x='weekday',
                          y=[('Concentration', 'mean'),
                              ('Concentration', 'median')],
                          rot=0, ax=ax4)
    ax4.legend(["mean", "median"])
    ax4.set(xlabel='weekday', ylabel="Concentration")

    seasonal_data.plot.bar(x='season',
                           y=[('Concentration', 'mean'),
                               ('Concentration', 'median')],
                           rot=0, ax=ax5)
    ax5.legend(["mean", "median"])
    ax5.set(xlabel='season', ylabel="Concentration")

    fig1.tight_layout()
    fig1.savefig('output\EDA_time.png')

    fig2, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 20))

    data_altitude.plot.bar(x='Altitude',
                           y=[('Concentration', 'mean'), ('Concentration', 'median')], rot=270, ax=ax1)
    ax1.legend(['mean', 'median'])
    ax1.set(xlabel='altitude', ylabel='Concentration')

    data_stationType.plot.bar(x='AirQualityStationType',
                              y=[('Concentration', 'mean'), ('Concentration', 'median')], rot=0, ax=ax2)
    ax2.legend(['mean', 'median'])
    ax2.set(xlabel='Station type', ylabel='Concentration')

    data_area.plot.bar(x='AirQualityStationArea',
                       y=[('Concentration', 'mean'), ('Concentration', 'median')], rot=0, ax=ax3)
    ax3.legend(['mean', 'median'])
    ax3.set(xlabel='Station Area', ylabel='Concentration')

    fig2.savefig('output\EDA_location.png')


def plot_timeseries(data):
    '''
    A FUNCTION that plots concentration evolution with time.
    INPUT: data
    OUTPUT: plot
    '''

    # variables to plot (values, time)
    time = mdates.date2num(data['DatetimeEnd'])
    values = data['Concentration']

    # Definition of parameters that will be used in the plot
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    days = mdates.DayLocator()
    # years format
    years_fmt = mdates.DateFormatter('%Y')
    months_fmt = mdates.DateFormatter(
        '%m-%Y')                                  # dates format

    # Plot
    fig, ax = plt.subplots()
    ax.plot_date(time, values, 'b-', label='max')

    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(days)
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.grid(False)
    ax.set(ylabel='Concentration (microgram/m3)')
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.legend()
    plt.figure(figsize=[15, 4])
    fig.savefig('output\hourly_concentration.png')


def make_new_dataset(save):
    '''
    A. Read datasets
        * list of EU country codes
        * list of pollutants
        * list of pollutants per country
    '''

    country_city_list = pd.read_json('data\country_city_list.json')
    # list of country codes
    EU_countries = country_city_list[0].unique()

    pollutants = pd.read_csv(
        'http://dd.eionet.europa.eu/vocabulary/aq/pollutant/csv',
        header=0, usecols=['URI', 'Notation'], index_col='URI')                 # list of pollutants [name, code]
    pollutants.index = pollutants.index.str.replace(r'\D', '')                  # replace the url with the code

    metadata = pd.read_csv(
        'https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv',
        sep='\t', usecols=['Countrycode', 'AirPollutantCode']).drop_duplicates()
    metadata.AirPollutantCode = metadata.AirPollutantCode.str.replace(r'\D', '')# replace the url with the code

    '''
    B. Take user input
        * Country : The user needs to choose a country in a list. By hitting ENTER
        they can choose all, but the dataset is too large and relatively large
        computer power.
        * City : The user needs to choose a city in the list or hit enter for all.
        * Pollutant : The user needs to choose a pollutant from the list of
        pollutants available for the country.
    '''
    country = choose_from_list(
        'Choose a Country Code :\n', EU_countries).upper()                      # Country

    # This option requires larger computer power.
    if (country == ''):

        city = ''
        pollutant = choose_from_list(
            'Choose a pollutant :\n', main_pollutants).upper()

    else:

        country_pollutants_codes = pd.Series(
            metadata[metadata.Countrycode == country].AirPollutantCode.unique())# List of pollutants codes for the country
        # For each pollutant code in the list of pollutant codes, map the notation.
        # The index of the pollutants dataframe coincides with the country-pollutants
        # codes.
        country_pollutants = country_pollutants_codes.map(pollutants['Notation'])# List of pollutants for the country

        # List of cities for the country
        cities_list = country_city_list[country_city_list[0] == country][1]

        city = choose_from_list('Choose a city or type ENTER for all :\n',
                                cities_list).capitalize()                       # City

        pollutant = choose_from_list('Choose a pollutant :\n',
                                     country_pollutants).upper()                # Pollutant

    pollutant_code = pollutants[pollutants.Notation == pollutant].index[0]      # Pollutant Code

    year1 = choose_from_list('Choose a year to start with:\n', years_list)      # Starting year
    year2 = choose_from_list('Choose a year to end with:\n', years_list)        # Ending year

    # Request the data to the url and count the time for the extraction
    start_time = time.perf_counter()
    dataset = get_data(country, city, pollutant_code, year1, year2)
    time_elapsed = time.perf_counter()-start_time
    print(f'It took {time_elapsed:.3f} seconds to extract the dataset.')

    if save :
        filename = 'data\\'+country+'_'+city+'_'+pollutant_code+'_'+year1+'_'+year2+'.csv'
        print('Saving the dataset to ',filename)
        dataset.to_csv(filename, index=False)

    return dataset


def add_AQindex(data):

    # Incomplete because I need to pass the pollutant as an argument
    thresholds = pollution_thresholds[pollution_thresholds.pollutant == pollutant].reset_index(
            drop=True)

    data['AQindex'] = ''

    data.loc[data.Concentration <= thresholds['Extremely Poor'][0], 'AQindex'] = 'Extremely Poor'
    data.loc[data.Concentration.between(thresholds.Moderate[0],
                                    thresholds.Poor[0]), 'AQindex'] = 'Poor'
    data.loc[data.Concentration.between(thresholds.Fair[0],
                                    thresholds.Moderate[0]), 'AQindex'] = 'Moderate'
    data.loc[data.Concentration.between(thresholds.Good[0],
                                    thresholds.Fair[0]), 'AQindex'] = 'Fair'
    data.loc[data.Concentration <=thresholds.Good[0], 'AQindex'] = 'Good'

    return data

def valid_date(s):
    try:
        return datetime.datetime.strptime(s,  "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def main():
    '''
    This code takes multiple input from the user and returns dataset.
    Datasets : country-city list, vocabulary of pollutants, PanEuropean_metadata
    User Input : Countrycode, Year, Pollutant
    Output : working_dataset
    '''
    parser = argparse.ArgumentParser(
        prog = 'pollution',
        description = '''
        ''',
        epilog = '''
        ''')


    parser.add_argument('-eda', '--eda', nargs = '?', default = False, const = True, type=bool,
                        help = 'Perform exploratory data analysis')

    # It is not allowed to parse a data file and save data in a data file. The
    # user can either parse an existing file with data or create a new dataset
    # and export it to a file with 'save'. For predictions the dataset is created
    # around the user input.
    group_data = parser.add_mutually_exclusive_group()
    group_data.add_argument('-df','--datafile', action='store',
                        nargs = '?', help = 'Parse a file with data')
    group_data.add_argument('-p', '--prediction', nargs = '?', default = False,
                        const = True, type=bool, help = 'Make a prediction')

    parser.add_argument('-ml', '--machine-learning', nargs = '?',
                        default = False, const = True, type = bool,
                        help = 'Perform machine-learning')
    parser.add_argument('-m', '--model', action = 'store', nargs = '?',
                        default = 'Decision Tree',
                        help='Choose a machine-learning model')

    parser.add_argument('-coord', '--coordinates', action = 'store', type = float,
                        nargs = 2, help='Get (lon, lat) for prediction')
    parser.add_argument('-d', '--date', action = 'store',
                        nargs = '?', default = datetime.date.today(),
                        type = valid_date, help='Get date for prediction')
    parser.add_argument('-h', '--hour', action = 'store',
                        nargs = '?', default = 9,
                        type = int, help='Get time for prediction')
    parser.add_argument('-pol', '--pollutant', action = 'store',
                        nargs = '?', choices = main_pollutants,
                        help='Get pollutant for prediction')

    parser.add_argument('-s', '--save', nargs = '?', default = False,
                        const = True, type=bool, help = 'Save dataset')

    args = parser.parse_args()

    if args.datafile : data = pd.read_csv(args.datafile)
    elif args.prediction:
        coord = args.coordinates
        date = args.date

        # Find cluster of local points
        coordinates_data, stations = get_coordinates_data(args.pollutant)
        station_ids = find_cluster(coordinates_data, coord, stations)
        data = get_data_for_prediction(args.pollutant, station_ids)

        # Get the new point for prediction
        if date.month.isin([12, 1, 2]):
            season = 'winter'
        elif date.month.isin([3, 4, 5]):
            season = 'spring'
        elif date.month.isin([6, 7, 8]):
            season = 'summer'
        else:
            season = 'autumn'

        new_data = {'Longitude':coord[0], 'Latitude':coord[1], 'year':date.year,
               'month':date.month, 'day':date.day, 'weekday':date.weekday(),
               'hour':args.hour, 'season':season}
        new = pd.DataFrame(new_data)

        # classification

    else : data = make_new_dataset(args.save)

    # Exploratory Data Analysis
    if args.eda:
         # Data preparation
        clean_dataset = clean_data(data)

        # Link data to coordinates
        data = link_data(clean_dataset, coordinates)

        print('The unit(s) of measurement is/are:')
        print(data['UnitOfMeasurement'].unique())
        print('The averaging time(s) for measurement is/are:')
        print(data['AveragingTime'].unique())
        print('The dataset includes :\n',','.join(data.columns))

        eda_columns = ['Concentration','DatetimeEnd','year','month','day','hour',
            'weekday','season','Longitude','Latitude','Altitude','AirQualityStationType',
            'AirQualityStationArea']
        print('For EDA we use ', eda_columns)
        EDA_pollution(data[eda_columns])

    if args.save :
        columns = ['year','month','day','hour','weekday','season','Longitude','Latitude','Altitude','Concentration']
        print('We keep the columns ',columns)
        filename = 'data\\prediction_'+args.pollutant+'.csv'
        print('Saving the dataset to ',filename)
        data[columns].to_csv(filename, index=False)

    if args.machine_learning:

        data = data[ml_variables]
        AQ_index = False
        if AQ_index : data = add_AQindex(data)                                  # Needs pollutant name

        # print(working_dataset[working_dataset['AQindex']=='Good'])
        # print(pollution_thresholds)
        #test_dataset = pd.DataFrame(data = {'Concentration' : 10}, index=(0,1))



if __name__ == '__main__':
    main()
