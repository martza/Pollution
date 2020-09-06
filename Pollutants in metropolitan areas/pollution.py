import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import sys
import os
import matplotlib.pyplot as plt

eea_url = 'https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw'
years_list = [str(x) for x in list(range(2013,2021))]
# maximum pollution thresholds for each caregory in microgram/m3
# taken from https://www.eea.europa.eu/themes/air/air-quality-index
pollution_thresholds = pd.DataFrame( data = {
    'pollutant' : ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2'],
    'Good' : ['10', '20', '40', '50', '100'],
    'Fair' : ['20', '40', '90', '100', '200'],
    'Moderate' : ['25', '50', '120', '130', '350'],
    'Poor' : ['50', '100', '230', '240', '500'],
    'Very Poor' : ['75', '150', '340', '380', '750'],
    'Extremely Poor' : ['800', '1200', '1000', '800', '1250']
}
)

def choose_from_list(text, list) :
    '''
    A FUNCTION that provides the list of options to the user and requires from
    the user to choose one of the options available.

    INPUT :
        * text : A message that requests action from the user
        * list : The list of options

    OUTPUT : value : The validated input from the user

    '''
    valid = False                                                               # Flags if the choice is valid
    while (not valid) :
        choice = input(text + ', '.join(list) + '\n')
        if choice.casefold() in map(str.casefold, list) :                       # Case insensitive search
            valid = True
        else :
            print('Invalid input. Choose an option from the list.')

    return choice


def get_data(countrycode, cityname, pollutantcode, year_from, year_to) :
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
                    "Pollutant" : pollutantcode, "Year_from" : year_from,
                    "Year_to": year_to, "Station":"", "Samplingpoint":"",
                    "Source":"E2a", "Output":"HTML", "UpdateDate":"",
                    "TimeCoverage":"Year"}
    response = requests.get(eea_url, params = query_params)

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
    dfs = (pd.read_csv(f,parse_dates = ['DatetimeBegin','DatetimeEnd'],
                       infer_datetime_format = True) for f in urls)
    data = pd.concat(dfs, ignore_index=True)

# Checking the characteristics of the dataset
    print('The dataset has the following column names:\n'+','.join(data.columns))
    print('The corresponding data types are:')
    print(data.dtypes)

    return data

def clean_data(data) :

    '''
    A FUNCTION that cleans the dataset from NAs and invalid points.

    INPUT :
        * data : The dataset

    OUTPUT :
        * A dataset with valid points
        * A visualisation of validity and verification points

    '''
    data['Concentration'] = data['Concentration'].fillna(0)

    # Check for NAs in Validity
    unvalid_points = pd.isna(data['Validity']).sum()
    print(f'There are {unvalid_points:.0f} points missing validation data.')

    # Check for invalid points
    validity_counts = data[['Concentration','Validity']].groupby(['Validity']).count()
    validity_counts = validity_counts.rename(columns = {'Concentration':'Counts'})

    # Check for NAs in Verification
    unverified_points = pd.isna(data['Verification']).sum()
    print(f'There are {unverified_points:.0f} points lacking verification data.')

    # Check for data that lack verification
    verification_counts = data[['Concentration','Verification']].groupby(['Verification']).count()
    verification_counts = verification_counts.rename(columns = {'Concentration':'Counts'})
    # Note that most of the data for 2019 and 2020 are not verified

    # Visualise verified and valid points
    fig, (ax1, ax2) = plt.subplots(2, figsize = (10,10))

    ax1.bar(verification_counts.index, 'Counts', data = verification_counts,
            label = 'Verification')
    ax1.set(xlabel = 'Verification', ylabel = 'Counts')
    ax1.legend()

    ax2.bar(validity_counts.index, 'Counts', data = validity_counts,
            label = 'Validity')
    ax2.set(xlabel = 'Validity', ylabel = 'Counts')
    ax2.legend()

    fig.savefig('check.png')

    # Remove invalid data, accept data that lack verification
    data = data[data['Validity']==1] # Getting rid of invalid data
    # Order data by datetime
    data = data.sort_values(by = 'DatetimeEnd')

    return data

'''
This code takes multiple input from the user and returns a heatmap of Pollution.
Datasets : country-city list, vocabulary of pollutants, PanEuropean_metadata
User Input : Countrycode, Year, Pollutant
Output : heatmap
'''

'''
A. Read datasets
    * list of EU country codes
    * list of pollutants
    * list of pollutants per country
'''

country_city_list = pd.read_json('country_city_list.json')
EU_countries = country_city_list[0].unique()                                    # list of country codes

pollutants = pd.read_csv(
    'http://dd.eionet.europa.eu/vocabulary/aq/pollutant/csv',
     header = 0, usecols = ['URI', 'Notation'], index_col = 'URI')                                 # list of pollutants [name, code]
pollutants.index = pollutants.index.str.replace(r'\D', '')                      # replace the url with the code

metadata = pd.read_csv(
    'https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv',
     sep = '\t', usecols = ['Countrycode', 'AirPollutantCode']).drop_duplicates()
metadata.AirPollutantCode = metadata.AirPollutantCode.str.replace(r'\D','')     # replace the url with the code

coordinates = pd.read_csv(
    'https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv',
     sep = '\t', usecols = ['AirQualityStationEoICode', 'Longitude', 'Latitude',
                            'Altitude']).drop_duplicates()                      # list of sampling points with local coordinates

'''
B. Take user input
    * Country
    * City
    * Pollutant
'''
country = choose_from_list('Choose a Country Code :\n', EU_countries).upper()   # Country
country_pollutants_codes = pd.Series(
    metadata[metadata.Countrycode == country].AirPollutantCode.unique())        # List of pollutants codes for the country
# For each pollutant code in the list of pollutant codes, map the notation.
# The index of the pollutants dataframe coincides with the country-pollutants
# codes.
country_pollutants = country_pollutants_codes.map(pollutants['Notation'])       # List of pollutants for the country

cities_list = country_city_list[country_city_list[0] == country][1]
cities_list = np.append(cities_list, 'all')

city = choose_from_list('Choose a city or type all :\n',
                        cities_list).capitalize()                               # City

pollutant = choose_from_list('Choose a pollutant :\n',
                              country_pollutants).upper()                       # Pollutant
pollutant_code = pollutants[pollutants.Notation == pollutant].index[0]          # Pollutant Code

year1 = choose_from_list('Choose a year to start with:\n', years_list)          # Starting year
year2 = choose_from_list('Choose a year to end with:\n', years_list)            # Ending year

# Request the data to the url and count the time for the extraction
start_time = time.perf_counter()
dataset = get_data(country, city, pollutant_code, year1, year2)
time_elapsed = time.perf_counter()-start_time
print(f'It took {time_elapsed:.3f} seconds to extract the dataset.')

# Data preparation
clean_dataset = clean_data(dataset)

#Exploratory analysis

print(clean_dataset['UnitOfMeasurement'].unique())
# Merge coordinates with pollution concentration
dataset_coordinates = pd.merge(clean_dataset, coordinates,
                               on = 'AirQualityStationEoICode')
print(dataset_coordinates.head())
print(pollution_thresholds)
