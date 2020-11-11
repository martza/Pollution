import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward

# Read metadata with Coordinates, pollutants, stations information
def get_coordinates_data(pollutant):
    coordinates = pd.read_csv(
            'https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv',
            sep='\t')
    coordinates.AirPollutantCode = coordinates.AirPollutantCode.str.replace(r'\D', '')
    # Keep the list of stations per country
    stations = coordinates[['Countrycode','AirQualityStationEoICode']].drop_duplicates()
    country_stations = stations.groupby(by=['Countrycode']).count()

    # Get mapping from pollutants to pollutant codes. Here for SO2.
    pollutants = pd.read_csv(
        'http://dd.eionet.europa.eu/vocabulary/aq/pollutant/csv',
        header=0, usecols=['URI', 'Notation'], index_col='URI')                 # list of pollutants [name, code]
    pollutants.index = pollutants.index.str.replace(r'\D', '')                  # replace the url with the code
    pollutant_code = pollutants[pollutants.Notation == pollutant].index[0]

    coordinates_data = coordinates[coordinates.AirPollutantCode == pollutant_code][[
        'AirQualityStationEoICode','Longitude', 'Latitude']]
    coordinates_data.drop_duplicates(inplace=True)

    return coordinates_data, stations

# include cluster information to coordinates data and make a prediction

def print_clusters_info(data):
    # some checks
    # Count the number of points per cluster
    count = data.groupby(by='Cluster').agg(Count = pd.NamedAgg(column='Longitude',
                                                             aggfunc='count'))
    count = count.sort_values(by = 'Count', ascending = False)

    idx_bigger = count.idxmax()
    print('The number of clusters in the dataset is:',len(count.index))
    print('The largest cluster is ', idx_bigger[0],'\n')
    print('The largest cluster contains ', count.max()[0],'points \n')

    # big_cluster = data[data.Cluster == idx_bigger[0]]

def find_cluster(locations, new, stations, method = 'agglomerative', linkage = 'complete', verbose = False):
    from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

    # Ensure that the clusters are equally distributed geographically by specifying
    # a distance threshold
    X = locations.drop('AirQualityStationEoICode',axis=1)

    if (method == 'kmeans'):
        clustering = KMeans(n_clusters = 100, n_init = 10, algorithm = 'auto').fit(X)
        if verbose:
            print('The cluster centers are: \n', clustering.cluster_centers_)
            print('The inertia is: \n', clustering.inertia_)
            print('The labels of the cluster are: \n', list(clustering.labels_))
    elif (method == 'dbscan'):
        clustering = DBSCAN(eps = 0.5, min_samples = 1, metric = 'euclidean',
            algorithm = 'auto').fit(X)
        if verbose:
            print('The indices of the core samples are:\n', clustering.core_sample_indices_)
            print('The components for each core sample are:\n', clustering.components_)
    else :
        clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory='cache',
                        connectivity=None, compute_full_tree='auto', linkage = linkage,
                        distance_threshold=2.0).fit(X)


    # Store the cluster index in new column and count the number of points per cluster.
    # Then find the cluster with the largest number of points just to check... To put
    # in a function.

    locations['Cluster'] = clustering.labels_
    print_clusters_info(locations)
    #linkage = ward(clustering.children_)

    #plt.title('Hierarchical Clustering Dendrogram')
    #dendrogram(linkage, truncate_mode='level', p=3)
    #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    #plt.show()

    # Take a new geographic location and add it to the dataset
    #new = np.array([4.7038946, 50.8864501])
    Y = np.vstack((X,new))
    # put the new location to a cluster
    cluster_labels = clustering.fit_predict(Y)

    # add the cluster labels to the dataset of coordinates
    Y_new = np.c_[Y,cluster_labels]

    df = pd.DataFrame(data =  Y_new, columns = ['Longitude','Latitude','Cluster'])
    df = df.sort_values(by='Cluster', ascending=True)
    print_clusters_info(df)

    # Get the cluster of the new point
    cluster = df.loc[(df.Longitude == new[0])&(df.Latitude == new[1])].Cluster.values[0]
    print('The new point is in the cluster number ', cluster)

    # Get the coordinates in the cluster and merge it with the airquality station information
    df1 = df[df.Cluster==cluster][['Longitude','Latitude']]
    df2 = pd.merge(df1, locations, how = 'inner', on = ['Longitude', 'Latitude'])
    print('There are ',len(df2.AirQualityStationEoICode.unique()),'stations in the cluster')
    station_ids = df2.AirQualityStationEoICode.unique()
    print('The stations are :\n',station_ids)
    countryCodes = stations[stations.AirQualityStationEoICode.isin(station_ids)].Countrycode.unique()
    print('The cluster includes the countries',countryCodes)

    return station_ids


#Get the data only fot the station ids

def get_data_for_prediction(pollutant, station_ids):
    # For each station id querry the data
    import requests
    from bs4 import BeautifulSoup
    eea_url = 'https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw'
    urls = []
    for id in station_ids:
        query_params = {"CountryCode": '', "Pollutant": pollutant, "Year_from": 2018,
            "Year_to": 2020, "EoICode": id,
            "Source": "E2a", "Output": "HTML", "TimeCoverage": "Year"}
        response = requests.get(eea_url, params=query_params)
    #       print('Getting data from:\n' + response.url)
    #       print('The headers of the response are:\n' + ','.join(response.headers))
    #       print('The response is:\n' + response.text)

        soup = BeautifulSoup(response.content, "lxml")
        for link in soup.find_all("a"):
            urls.append(link.get("href"))
    print('Found ',len(urls),' urls')
    dfs = (pd.read_csv(f, parse_dates=['DatetimeBegin', 'DatetimeEnd'],
                infer_datetime_format=True) for f in urls)

    # This takes some time
    data = pd.concat(dfs, ignore_index=True)

    print('Got data from the following stations:\n',
        data.AirQualityStationEoICode.unique())
    return data

def calibration(data, stations):
    for station in stations:
        data_station = data[data.AirQualityStation == station]
        background = -data_station.Concentration.min()
        new_concentration = data_station.Concentration + background
        data.loc[data.AirQualityStation == station,'Concentration'] = new_concentration
    return data

# Find nearest neighbors
#### Testing
if __name__ == '__main__':
    new = [4.7038946, 50.8864501]
    coordinates_data, stations = get_coordinates_data('SO2')

    test_clustering = False
    clustering = 'agglomerative'
    save_data = False
    get_data = True

    if test_clustering:
        if (clustering == 'kmeans' or clustering == 'all'):
            print('TRYING K-Means CLUSTERING...')
            print(coordinates_data.columns)
            station_ids = find_cluster(coordinates_data, new, stations,
                method = 'kmeans')
            coordinates_data = coordinates_data.drop('Cluster', axis = 1)
        elif (clustering == 'agglomerative' or clustering == 'all'):
            print(coordinates_data.columns)
            print('TRYING Agglomerative CLUSTERING...')
            station_ids = find_cluster(coordinates_data, new, stations)
            coordinates_data = coordinates_data.drop('Cluster', axis = 1)
        elif (clustering == 'dbscan' or clustering == 'all'):
            print(coordinates_data.columns)
            print('TRYING DBSCAN CLUSTERING...')
            station_ids = find_cluster(coordinates_data, new, stations,
                method = 'dbscan', verbose = True)

    if save_data:
        data = get_data_for_prediction('SO2', station_ids)
        filename = 'data\\prediction_SO2.csv'
        print('Saving the dataset to ',filename)
        data.to_csv(filename, index=False)

    if get_data:
        data = pd.read_csv('data\\prediction_SO2.csv',
            parse_dates=['DatetimeBegin', 'DatetimeEnd'], infer_datetime_format = True)
        print(data.columns)
        print(data.info)
        print('There are', sum(data.Validity == -1),' invalid points')
        print('Removing invalid points...')
        data = data[data.Validity == 1]
        print('There are ',sum(data.Concentration.isna()),'missing values in Concentration')

        # Add to the dataset year, day, month, hour, weekday columns
        data['year'] = pd.DatetimeIndex(data['DatetimeEnd']).year
        data['month'] = pd.DatetimeIndex(data['DatetimeEnd']).month
        data['day'] = pd.DatetimeIndex(data['DatetimeEnd']).day
        data['hour'] = pd.DatetimeIndex(data['DatetimeEnd']).hour
        data['weekday'] = data['DatetimeEnd'].dt.dayofweek

        data[data.Verification == 3].year.unique()
        data.year.unique()
        provisional = sum(data.Verification == 3)
        preliminary = sum(data.Verification == 2)
        verified = sum(data.Verification == 1)
        total = data.shape[0]
        print(provisional/total,' of the data are provisional.')
        print(preliminary/total,' of the data are preliminary.')
        print(verified/total,' of the data are verified.')

        data.DatetimeEnd[0].to_pydatetime().month

        # Getting read of negative values.
        stations_to_calibrate2 = data.loc[(data.Concentration < 0) & (data.Verification == 2),'AirQualityStation'].unique()
        data1 = calibration(data, stations_to_calibrate2)
        stations_to_calibrate3 = data1.loc[(data1.Concentration < 0) & (data1.Verification == 3),'AirQualityStation'].unique()
        data2 = calibration(data1, stations_to_calibrate3)

        negatives = sum(data2.Concentration < 0)
        print('There are ', negatives, ' negative Concentation points in the dataset.')
        print('The maximum Concentration is ', data2.Concentration.max())
        print('The minimum Concentration is ', data2.Concentration.min())

        # For the station I need to augment all measurements by the minimum.