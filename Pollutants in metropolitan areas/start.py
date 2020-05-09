import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

################################################################################
#Query and load the data
################################################################################
#Query
response = requests.get('https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw?CountryCode=BE&CityName=Bruxelles%20/%20Brussel&Pollutant=8&Year_from=2019&Year_to=2020&Station=&Samplingpoint=&Source=E2a&Output=HTML&UpdateDate=&TimeCoverage=Year')
#The request can be optimized further

#Some commands to check the properties of the response:
#response.headers  #1
#response.encoding  #2
#response.text  #3

#Use BeautifulSoup(kwargs) to parse urls from the response into a list
soup = BeautifulSoup(response.content, "lxml")

#Construct a list of urls only from the 'href' field
urls = []
for link in soup.find_all("a"):
    urls.append(link.get("href"))

#Read from all urls in the list and parse to a dataframe
dfs = (pd.read_csv(f,parse_dates = ['DatetimeBegin','DatetimeEnd'], infer_datetime_format = True) for f in urls)
data = pd.concat(dfs, ignore_index=True)
data.columns
data.dtypes
################################################################################
#cleaning the data
#Concentration is the values column (y).
#Validity and Verification columns indicate the quality of the data
################################################################################
#Remove NAs in y
data['Concentration'] = data['Concentration'].fillna(0)
#Check for NAs in Validity
pd.isna(data['Validity']).sum()
#Check for invalid points
validity_counts = data[['Concentration','Validity']].groupby(['Validity']).count()
validity_counts = validity_counts.rename(columns = {'Concentration':'Counts'})
validity_counts.plot.bar()
#Check for NAs in Verification
pd.isna(data['Verification']).sum()
#Check for data that lack verification
verification_counts = data[['Concentration','Verification']].groupby(['Verification']).count()
verification_counts = verification_counts.rename(columns = {'Concentration':'Counts'})
verification_counts.plot.bar() # Note that most of the data for 2019 and 2020 are not verified
#Remove invalid data, accept data that lack verification
data = data[data['Validity']==1] # Getting rid of invalid data
#Order data by datetime
data = data.sort_values(by = 'DatetimeEnd')

################################################################################
#Question: How is NO2 concentration evolving in metropolitan areas?
#Is the NO2 concentration in acceptable levels where most people live?
#What are the factors that affect the measurements of the NO2 concentation in metropolitan areas?
################################################################################
#Sampling points in metropolitan areas give the 'background data'
#Get the list of stations used for collecting background data
Brussels_stations = pd.read_csv('Brussels_stations_coordinates')
Brussels_stations.columns
list = Brussels_stations['AirQualityStation']
#Keep NO2 data that are only background data
background_data = data[data['AirQualityStation'].isin(list.to_list())]
background_data.reset_index(inplace = True, drop = True)
#Merge background data with station coordinates
background_data = pd.merge(background_data, Brussels_stations, on = 'AirQualityStation' )

#At each datetime point, measurements are taken from different Brussels stations.
#We want to see the impact of altitude on the NO2 measurements.
#For this purpose we group by altitude.
background_data2 = background_data[['Concentration','Altitude']].groupby(['Altitude'], as_index = False).aggregate([np.mean,np.median]).reset_index()
fig0, ax0 = plt.subplots()
background_data2.plot.bar(x = 'Altitude', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax0)
ax0.legend(["mean", "median"])
ax0.set_ylabel("NO2 Concentration")

#We will keep the lower altitudes <=20 m. , because it is where most of the people live.

background_data = background_data[background_data['Altitude']<=20]
background_data[['Concentration','DatetimeEnd']].groupby(['DatetimeEnd'], as_index = False).count().max()
#There is maximum 2 points per hour. For the worst-case scenario we take the maximum.
background_data_max = background_data[['Concentration','DatetimeEnd']].groupby(['DatetimeEnd'], as_index = False).max().reset_index()
background_data_max.columns

time = mdates.date2num(background_data_max['DatetimeEnd'])
max = background_data_max['Concentration']
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
days = mdates.DayLocator()
years_fmt = mdates.DateFormatter('%Y')
months_fmt = mdates.DateFormatter('%m-%Y')

#Plot of NO2 concentration evolution over the years 2019 and 2020
fig, ax = plt.subplots()
ax.plot_date(time, max, 'b-', label = 'max')

# format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
ax.xaxis.set_minor_locator(days)

# round to nearest years.
#datemin = min(background_data_max['DatetimeEnd'])
#datemax = max(background_data_max['DatetimeEnd'])
#ax.set_xlim(datemin, datemax)
# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.grid(False)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
plt.legend()
plt.figure(figsize = [15,4])
plt.show()

#In the metropolitan Brussels area the hourly NO2 measurements do not exceed
#the acceptable limits of 200

#Some statistics on the dataset:
background_data_max['Concentration'].plot.hist()
background_data_max['Concentration'].describe()


#Now we add to the dataset year, day, month, hour, weekday columns
background_data_max['year'] = pd.DatetimeIndex(background_data_max['DatetimeEnd']).year
background_data_max['month'] = pd.DatetimeIndex(background_data_max['DatetimeEnd']).month
background_data_max['day'] = pd.DatetimeIndex(background_data_max['DatetimeEnd']).day
background_data_max['hour'] = pd.DatetimeIndex(background_data_max['DatetimeEnd']).hour
background_data_max['weekday'] = background_data_max['DatetimeEnd'].dt.dayofweek

#We aggregate the NO2 concentration per year and per month as a mean and compare years 2019 and 2020
NO2_per_year = background_data_max[background_data_max['month']<6][['Concentration','year']].groupby(['year']).mean()
fig4, ax4 = plt.subplots()
NO2_per_year['Concentration'].plot.bar()
ax4.legend(["mean"])
ax4.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_year.png')

#For the first four months of 2019 and 2020 the mean concentration of NO2 presents a drop of 30%.
#This can be interpreted as an effect of the COVID-19 crisis in brussels

NO2_per_month = background_data_max[['Concentration','year','month']].groupby(['year','month']).mean()
NO2_per_month['Concentration'].plot.bar()
#The biggest drop can be seen in the month of February, where the NO2 concentration fell in half the one of 2019

#Explore the evolution of the concentration of NO2 with the hour of the day
NO2_per_hour = background_data_max[['DatetimeEnd','Concentration','hour']].groupby(['hour'], as_index = False).aggregate([np.mean, np.median]).reset_index()
NO2_per_hour.columns
fig1, ax1 = plt.subplots()
NO2_per_hour.plot.bar(x = 'hour', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax1)
ax1.legend(["mean", "median"])
ax1.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_hour.png')
#The graph shows the evolution of the concentration with the hour of the day.
#There is a pattern, which is probably related to the working habbits of people.

#Explore the evolution of the concentration of NO2 with the weekday
NO2_per_weekday = background_data_max[['DatetimeEnd','Concentration','weekday']].groupby(['weekday'], as_index = False).aggregate([np.mean, np.median]).reset_index()
fig2, ax2 = plt.subplots()
NO2_per_weekday.plot.bar(x = 'weekday', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax2)
ax2.legend(["mean", "median"])
ax2.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_weekday.png')
# There is a drop of the concentration of NO2 on Saturdays and Sundays

# Define a new categorical variable for the seasons
background_data_max['season'] = ''
background_data_max.loc[background_data_max['month'].isin([12,1,2]),'season'] = 'winter'
background_data_max.loc[background_data_max['month'].isin([3,4,5]),'season'] = 'spring'
background_data_max.loc[background_data_max['month'].isin([6,7,8]),'season'] = 'summer'
background_data_max.loc[background_data_max['month'].isin([9,10,11]),'season'] = 'autumn'
background_data_max.to_csv('NO2_Brussels.csv',index = False)

NO2_per_season = background_data_max[['DatetimeEnd','Concentration','season']].groupby(['season'], as_index = False).aggregate([np.mean, np.median]).reset_index()
fig3, ax3 = plt.subplots()
NO2_per_season.plot.bar(x = 'season', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax3)
ax3.legend(["mean", "median"])
ax3.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_season.png')
#There is a drop of the concentration of NO2 the summer and an increase in the winter.

################################################################################
#Additional variables that can affect the concentration of NO2 are: season,
#hour of the day, day of the week, social and political factors
################################################################################
