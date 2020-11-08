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

    big_cluster = data[data.Cluster == idx_bigger[0]]

def find_cluster(data, new, stations):
    from sklearn.cluster import AgglomerativeClustering

    # Ensure that the clusters are equally distributed geographically by specifying
    # a distance threshold
    X = data.drop('AirQualityStationEoICode',axis=1)

    clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory='cache',
                        connectivity=None, compute_full_tree='auto', linkage='complete',
                        distance_threshold=2.0).fit(X)

    #linkage = ward(clustering.children_)

    #plt.title('Hierarchical Clustering Dendrogram')
    #dendrogram(linkage, truncate_mode='level', p=3)
    #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    #plt.show()

    # Store the cluster index in new column and count the number of points per cluster.
    # Then find the cluster with the largest number of points just to check... To put
    # in a function.

    data['Cluster'] = clustering.labels_
    print_clusters_info(data)

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
    df2 = pd.merge(df1, data, how = 'inner', on = ['Longitude', 'Latitude'])
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

# Find nearest neighbors
#### Testing
if __name__ == '__main__':
    main()

