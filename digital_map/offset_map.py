import pandas as pd

####################################################
## This script applies an offset to every coordinate
## in the given dataset (needs to be used for map
## gathered by BAYSIS Version May2020 to August2020)
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################

def apply_offset(df_map):
    """
    This function applies a predefined offset to the coordinates
    because of offset in baysis dataset
    Example:        lat       lon
        Baysis:     48.235636 11.687087
        Original:   48.234668 11.685693
        Diff:       -0.000968 -0.001394


    :param df_map: Pandas DataFrame with all map data matched
    :return: df_map, same dataframe with corrected coordinates
    """

    diff_lon = -0.001399
    diff_lat = -0.000951

    df_map.longitude = df_map.longitude.add(diff_lon)
    df_map.latitude = df_map.latitude.add(diff_lat)

    return df_map
