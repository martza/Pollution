import pandas as pd
import numpy as np
import gmaps
import gmaps.datasets
import os

gmaps.configure(api_key="AIzaSyBqmnTJMdfnOg0eE2FRUKAWE4qaqmNguoY")
data = pd.read_csv('NO2_Brussels.csv')
data.columns
locations = data[['Latitude','Longitude']]  
weights = data['Concentration']
fig = gmaps.figure()
heatmap_layer = gmaps.heatmap_layer(locations, weights=weights)
fig.add_layer(gmaps.heatmap_layer(locations, weights=weights))
fig
