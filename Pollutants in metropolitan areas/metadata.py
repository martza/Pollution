import pandas as pd
metadata = pd.read_csv('https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv', sep = '\t')
metadata.columns
metadata.AirPollutantCode = metadata.AirPollutantCode.str.replace(r'\D','')
metadata.AirPollutantCode
metadata.AirQualityNetwork.unique()
Brussels_metadata = metadata[metadata['Countrycode']=='BE'][metadata['AirQualityNetwork']=='NET-Brussels']
Brussels_metadata.columns
Brussels_stations = Brussels_metadata[['AirQualityStation','AirQualityStationType','AirQualityStationArea','Longitude','Latitude','Altitude']].groupby(['AirQualityStation'], as_index = False).last()
Brussels_stations.groupby(['AirQualityStationType'], as_index = False).count()
Brussels_stations.groupby(['AirQualityStationArea'], as_index = False).count()

Brussels_stations[Brussels_stations['AirQualityStationType']=='background'][['AirQualityStation','Longitude','Latitude','Altitude']].to_csv('Brussels_stations_coordinates',index = False)
#one station can have more than one sampling points and one location can have more than one stations.
