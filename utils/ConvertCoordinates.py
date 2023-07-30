import numpy as np
import pandas as pd
import geopandas as gpd


#####################################################
## This script converts to and from various
## coordinates (Gauss-Krueger, WGS84, UTM)
#####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
#####################################################
## Date: 06.2020
#####################################################


def convert_krueger_to_wgs84(east: np.ndarray, north: np.ndarray, from_epsg: int, to_epsg: int):
    """"Calculate WGS84 coordinates from Krueger coordinates

    In/Out: Dataframe with Krueger(GK4) or WGS84  or ETRS89/UTM(Zone32) coordinates
    
    east/north Input: each as ndarray (numpy)
    
    Output: pandas dataframe in order east|north

    Example:
        | Rechtswert|   Hochwert|
        |4512938.176|5379282.978|
        |4513870.070|5377491.353|
        ->
        |latitude    |  longitude|
        |48.55113141 |12.17379836|
        |48.53500014 |12.18636171|
    """
    # Bounding Box GPS
    # BBOX_Max = np.float64([[30.0, 80.0],[3.0, 18.0]])

    # Check instance type
    if isinstance(east, str) or isinstance(north, str):
        print("String detected in coordinates (probably empty)!")
        print('Values detected: ' + str(df_input.iloc[0, 0]) + ' and ' + str(df_input.iloc[0, 1]))
        return float('nan')
    if type(east) == type([]):
        east = np.asarray(east)
        north = np.asarray(north)
    if (east == float('nan')).any() or (north == float('nan')).any():
        print("nan detected in coordinates!")
        print('Values detected: ' + str(df_input.iloc[0, 0]) + ' and ' + str(df_input.iloc[0, 1]))
        return float('nan')

    # format: [longitude/East/Rechtswert, latitude/North/Hochwert]
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(east, north))
    # known formats:
    kruger_code = 'epsg:31468'
    wsg84_code = 'epsg:4258'
    utm_code = 'epsg:25832'  # ETRS89

    if from_epsg == to_epsg:
        print("Warning: Coordinate CRS from input and output identical. No need for coordinate convertion.")
    # Input
    if from_epsg == 31468:
        gdf.crs = {'init': kruger_code}
    elif from_epsg == 4258:
        gdf.crs = {'init': wsg84_code}
    elif from_epsg == 25832:
        gdf.crs = {'init': utm_code}
    # Output
    first_string = 'east'
    second_string = 'north'
    if to_epsg == 31468:
        gdf_new = gdf.to_crs({'init': kruger_code})
    elif to_epsg == 4258:
        gdf_new = gdf.to_crs({'init': wsg84_code})
        first_string = 'longitude'
        second_string = 'latitude'
    elif to_epsg == 25832:
        gdf_new = gdf.to_crs({'init': utm_code})

    first_value = [float(value) for value in gdf_new.geometry.x]
    second_value = [float(value) for value in gdf_new.geometry.y]

    df_output = pd.DataFrame({first_string: first_value, second_string: second_value})
    return df_output
