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
#What are the possible variables of the NO2 concentation in metropolitan areas?
################################################################################
#Sampling points in metropolitan areas give the 'background data'
#Get the list of stations used for collecting background data
Brussels_stations = pd.read_csv('Brussels_stations_coordinates')
Brussels_stations.columns
list = Brussels_stations['AirQualityStation']
#Keep NO2 data that are only background data
background_data = data[data['AirQualityStation'].isin(list.to_list())]
background_data.reset_index(inplace = True, drop = True)
#Merge background data with local coordinates
background_data = pd.merge(background_data, Brussels_stations, on = 'AirQualityStation' )



#At each datetime point, measurements are taken from different Brussels stations.
#We group by End date. Low measurements can spoil an average.
#Therefore we take the median of the measurements.

background_data1 = background_data[['Concentration','DatetimeEnd']].groupby(['DatetimeEnd'], as_index = False).aggregate([np.mean,np.median]).reset_index()
#exploratory analysis
background_data1.columns
#Definition of the variables
mean = background_data1[('Concentration','mean')]
median = background_data1[('Concentration','median')]
time = mdates.date2num(background_data1['DatetimeEnd'])
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
days = mdates.DayLocator()
years_fmt = mdates.DateFormatter('%Y')
months_fmt = mdates.DateFormatter('%m-%Y')

#Plot of NO2 concentration evolution over the years 2019 and 2020
fig, ax = plt.subplots()
y1 = ax.plot_date(time, median, 'b-')
y2 = ax.plot_date(time, median-mean, 'r-')

# format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
ax.xaxis.set_minor_locator(days)

# round to nearest years.
datemin = min(background_data1['DatetimeEnd'])
datemax = max(background_data1['DatetimeEnd'])
ax.set_xlim(datemin, datemax)
# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
#ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax.grid(False)
ax.plot_date(time, median, 'b-', label = 'Median')
ax.plot_date(time, median-mean, 'r-', label = 'Mean')


# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
plt.figure(figsize = [15,4])
plt.legend([y1,y2],['Median','Median-Mean'])
plt.show()











median_background_data_NO2['year'] = pd.DatetimeIndex(median_background_data_NO2['DatetimeEnd']).year
median_background_data_NO2['month'] = pd.DatetimeIndex(median_background_data_NO2['DatetimeEnd']).month
median_background_data_NO2['day'] = pd.DatetimeIndex(median_background_data_NO2['DatetimeEnd']).day
median_background_data_NO2['hour'] = pd.DatetimeIndex(median_background_data_NO2['DatetimeEnd']).hour

median_background_data_NO2['weekday'] = median_background_data_NO2['DatetimeEnd'].dt.dayofweek
#median_background_data_NO2.columns



median_background_data_NO2['Concentration'].plot.hist()
median_background_data_NO2['Concentration'].describe()

# Here I aggregate the NO2 concentration per year and per month as a mean and compare amongst years 2019 and 2020
NO2_per_year = median_background_data_NO2[median_background_data_NO2['month']<5][['Concentration','year']].groupby(['year']).mean() #Comparison between 2019 and 2020
fig4, ax4 = plt.subplots()
NO2_per_year['Concentration'].plot.bar()
ax4.legend(["mean"])
ax4.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_year.png')
#For the first four months of 2019 and 2020 the mean concentration of NO2 presents a drop of 30% and this can be seen as an effect of the COVID-19 crisis in belgium
NO2_per_month = median_background_data_NO2[['Concentration','year','month']].groupby(['year','month']).mean()
NO2_per_month['Concentration'].plot.bar()

# Define categorical variables: a. Seasons b. Weekdays [Monday-Sunday] d. Is public Holiday [True, False]
median_background_data_NO2['season'] = ''
median_background_data_NO2.loc[median_background_data_NO2['month'].isin([12,1,2]),'season'] = 'winter'
median_background_data_NO2.loc[median_background_data_NO2['month'].isin([3,4,5]),'season'] = 'spring'
median_background_data_NO2.loc[median_background_data_NO2['month'].isin([6,7,8]),'season'] = 'summer'
median_background_data_NO2.loc[median_background_data_NO2['month'].isin([9,10,11]),'season'] = 'autumn'
median_background_data_NO2.to_csv('NO2_Brussels.csv',index = False)

NO2_per_hour = median_background_data_NO2[['DatetimeEnd','Concentration','hour']].groupby(['hour'], as_index = False).aggregate([np.mean, np.median]).reset_index()
NO2_per_hour.columns
fig1, ax1 = plt.subplots()
NO2_per_hour.plot.bar(x = 'hour', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax1)
ax1.legend(["mean", "median"])
ax1.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_hour.png')
# There is a certain behaviour of the concentration regarding the time of the day
#NO2_per_hour
NO2_per_weekday = median_background_data_NO2[['DatetimeEnd','Concentration','weekday']].groupby(['weekday'], as_index = False).aggregate([np.mean, np.median]).reset_index()
fig2, ax2 = plt.subplots()
NO2_per_weekday.plot.bar(x = 'weekday', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax2)
ax2.legend(["mean", "median"])
ax2.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_weekday.png')

# As expected the cencentration on saturdays and Sundays is falling
NO2_per_season = median_background_data_NO2[['DatetimeEnd','Concentration','season']].groupby(['season'], as_index = False).aggregate([np.mean, np.median]).reset_index()
fig3, ax3 = plt.subplots()
NO2_per_season.plot.bar(x = 'season', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax3)
ax3.legend(["mean", "median"])
ax3.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_season.png')
#As expected NO2 falls in the summer and is increasing during the winter.
