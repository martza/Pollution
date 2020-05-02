import pandas as pd
import requests
#import re  #Uncomment for stripping
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

response = requests.get('https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw?CountryCode=BE&CityName=Bruxelles%20/%20Brussel&Pollutant=8&Year_from=2019&Year_to=2020&Station=&Samplingpoint=&Source=E2a&Output=HTML&UpdateDate=&TimeCoverage=Year')
#The request can be optimized further

#Some commands to check the properties of the response:
#response.headers  #1
#response.encoding  #2
#response.text  #3

#Use BeautifulSoup(kwargs) to parse urls from the response into a list
#For more, see https://www.pluralsight.com/guides/extracting-data-html-beautifulsoup
soup = BeautifulSoup(response.content, "lxml")
soup

#for link in soup.find_all("a"):
#    print("Inner Text: {}".format(link.text))
#    print("download: {}".format(link.get("title")))
#    print("href: {}".format(link.get("href")))

#For the urls we only need the 'href' field
urls = []
for link in soup.find_all("a"):
    urls.append(link.get("href"))

print(urls)
#urls[0]

#Some stripping commands (not used here):
#stripped = re.sub('ï»¿', '', response.text)  #1
#stripped1 = re.sub('https','http',stripped)  #2
#print(stripped1) #2

#Parse list of urls to a dataframe
dfs = (pd.read_csv(f,parse_dates = ['DatetimeBegin','DatetimeEnd'], infer_datetime_format = True) for f in urls)
data = pd.concat(dfs, ignore_index=True)
#data
data.columns
data.dtypes
data['DatetimeBegin']

#cleaning the data
data['Concentration'] = data['Concentration'].fillna(0)
pd.isna(data['Validity']).sum()
validity_counts = data[['Concentration','Validity']].groupby(['Validity']).count()
validity_counts = validity_counts.rename(columns = {'Concentration':'Counts'})
validity_counts.plot.bar()

pd.isna(data['Verification']).sum()
verification_counts = data[['Concentration','Verification']].groupby(['Verification']).count()
verification_counts = verification_counts.rename(columns = {'Concentration':'Counts'})
verification_counts.plot.bar() # Note that most of the data for 2019 and 2020 are not verified
data1 = data[data['Validity']==1] # Getting rid of invalid data
data1.columns

data1 = data1.sort_values(by = 'DatetimeEnd')

# At each datetime point, measurements are taken from different sampling points.
# We group by End date. Low measurements can spoil an average.
#Therefore we take the median of the measurements.

data_NO2 = data1[['Concentration','DatetimeEnd']].groupby(['DatetimeEnd'], as_index = False).median()
data_NO2
data_NO2['year'] = pd.DatetimeIndex(data_NO2['DatetimeEnd']).year
data_NO2['month'] = pd.DatetimeIndex(data_NO2['DatetimeEnd']).month
data_NO2['day'] = pd.DatetimeIndex(data_NO2['DatetimeEnd']).day
data_NO2['hour'] = pd.DatetimeIndex(data_NO2['DatetimeEnd']).hour

data_NO2['weekday'] = data_NO2['DatetimeEnd'].dt.dayofweek
data_NO2.columns

#exploratory analysis

#Definition of the variables
y = data_NO2['Concentration']
x = mdates.date2num(data_NO2['DatetimeEnd'])
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
days = mdates.DayLocator()
years_fmt = mdates.DateFormatter('%Y')
months_fmt = mdates.DateFormatter('%m-%Y')

#Plot of NO2 concentration evolution over the years 2019 and 2020
fig, ax = plt.subplots()
ax.plot_date(x, y, 'bo')

# format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
ax.xaxis.set_minor_locator(days)

# round to nearest years.
datemin = min(data_NO2['DatetimeEnd'])
datemax = max(data_NO2['DatetimeEnd'])
ax.set_xlim(datemin, datemax)
# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax.grid(False)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
plt.figure(figsize = [10,5])
plt.show()


data_NO2['Concentration'].plot.hist()
data_NO2['Concentration'].describe()

# Here I aggregate the NO2 concentration per year and per month as a mean and compare amongst years 2019 and 2020
NO2_per_year = data_NO2[data_NO2['month']<5][['Concentration','year']].groupby(['year']).mean() #Comparison between 2019 and 2020
fig4, ax4 = plt.subplots()
NO2_per_year['Concentration'].plot.bar()
ax4.legend(["mean"])
ax4.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_year.png')
#For the first four months of 2019 and 2020 the mean concentration of NO2 presents a drop of 30% and this can be seen as an effect of the COVID-19 crisis in belgium
NO2_per_month = data_NO2[['Concentration','year','month']].groupby(['year','month']).mean()
NO2_per_month['Concentration'].plot.bar()

# Define categorical variables: a. Seasons b. Weekdays [Monday-Sunday] d. Is public Holiday [True, False]
data_NO2['season'] = ''
data_NO2.loc[data_NO2['month'].isin([12,1,2]),'season'] = 'winter'
data_NO2.loc[data_NO2['month'].isin([3,4,5]),'season'] = 'spring'
data_NO2.loc[data_NO2['month'].isin([6,7,8]),'season'] = 'summer'
data_NO2.loc[data_NO2['month'].isin([9,10,11]),'season'] = 'autumn'
data_NO2
data_NO2.to_csv('NO2_Brussels.csv',index = False)

NO2_per_hour = data_NO2[['DatetimeEnd','Concentration','hour']].groupby(['hour'], as_index = False).aggregate([np.mean, np.median]).reset_index()
NO2_per_hour.columns
fig1, ax1 = plt.subplots()
NO2_per_hour.plot.bar(x = 'hour', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax1)
ax1.legend(["mean", "median"])
ax1.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_hour.png')
# There is a certain behaviour of the concentration regarding the time of the day
#NO2_per_hour
NO2_per_weekday = data_NO2[['DatetimeEnd','Concentration','weekday']].groupby(['weekday'], as_index = False).aggregate([np.mean, np.median]).reset_index()
fig2, ax2 = plt.subplots()
NO2_per_weekday.plot.bar(x = 'weekday', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax2)
ax2.legend(["mean", "median"])
ax2.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_weekday.png')

# As expected the cencentration on saturdays and Sundays is falling
NO2_per_season = data_NO2[['DatetimeEnd','Concentration','season']].groupby(['season'], as_index = False).aggregate([np.mean, np.median]).reset_index()
fig3, ax3 = plt.subplots()
NO2_per_season.plot.bar(x = 'season', y = [('Concentration','mean'),('Concentration','median')], rot = 0, ax = ax3)
ax3.legend(["mean", "median"])
ax3.set_ylabel("NO2 Concentration")
plt.savefig('NO2_Brussels_per_season.png')
#As expected NO2 falls in the summer and is increasing during the winter.
