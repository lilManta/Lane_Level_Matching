import pickle
import matplotlib.pyplot as plt
import mplleaflet
import numpy as np
import pandas as pd
from utils import ConvertCoordinates as convert
# from progressbar import ProgressBar  # since finding the closest IDs can take some time
import config.paths as paths

# pbar = ProgressBar()


#####################################################
## This script takes the lanes from the BAYSIS lanes
## Dataframe and determines the center of the roads
#####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
#####################################################
## Date: 05.2019
#####################################################


def addKruegerCoordinatesToCenterRoads(centerRoads, pickle_save: str, flattened=False, check_coordinates=False):
    """
        This function converts all center Road coordinates to Krueger coordinates (saves them in new columns)
        and then deletes all rows which have improper Krueger coordinates
    :param pickle_save: full path to save dataframe as pickle
    :param centerRoads: Dataframe to add the krueger coordinates to
    :return: None
    """

    if check_coordinates:
        # Get rid of rows with improper coordinates
        try:
            for index in centerRoads.index:
                if isinstance(centerRoads.at[index, 'centerRoadNorth'], str) or \
                        isinstance(centerRoads.at[index, 'centerRoadEast'], str):
                    if centerRoads.at[index, 'centerRoadNorth'] == '' and centerRoads.at[index, 'centerRoadEast'] == '':
                        centerRoads.drop(index=index, axis=0, inplace=True)
                    else:
                        print("Error")
                elif len(centerRoads.at[index, 'centerRoadNorth']) < 1 or len(centerRoads.at[index, 'centerRoadEast']) < 1:
                    centerRoads.drop(index=index, axis=0, inplace=True)
        except:
            print("Error while trying to extend center roads with Krueger coordinates.")

    centerRoads['centerRoadEastKrueger'] = ''
    centerRoads['centerRoadNorthKrueger'] = ''
    if not flattened:
        for index in centerRoads.index:
            # wgs84 format: [longitude/East, latitude/North]
            new_center = convert.convert_krueger_to_wgs84(east=centerRoads.loc[index, 'centerRoadEast'],
                                                          north=centerRoads.loc[index, 'centerRoadNorth'],
                                                          from_epsg=4258, to_epsg=31468)

            centerRoads.at[index, 'centerRoadEastKrueger'] = new_center.loc[:, 'east'].values
            centerRoads.at[index, 'centerRoadNorthKrueger'] = new_center.loc[:, 'north'].values
    else:
        new_center = convert.convert_krueger_to_wgs84(east=centerRoads.loc[:, 'centerRoadEast'],
                                                      north=centerRoads.loc[:, 'centerRoadNorth'],
                                                      from_epsg=4258, to_epsg=31468)

        centerRoads.loc[:, 'centerRoadEastKrueger'] = new_center.loc[:, 'east']
        centerRoads.at[:, 'centerRoadNorthKrueger'] = new_center.loc[:, 'north']

    # save the dataframe
    if pickle_save is not None:
        centerRoads.to_pickle(pickle_save)
