import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import mplleaflet
# import time
# from progressbar import ProgressBar
# pbar = ProgressBar()

import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.DetermineDistance import directionCalculation, get_orient_diff

####################################################
## This script is calculating and matching the
## orientation of the gps points to the street.
## It belongs to simple_map_match.py
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 07.2020
####################################################


def simple_orient_match(k_nearest: pd.DataFrame, start_gps: pd.Series,
                        end_gps: pd.Series, last_dir_bool: bool) -> pd.DataFrame:
    plot_errors = False

    gps_orientation = directionCalculation(start_gps.latitude, start_gps.longitude,
                                           end_gps.latitude, end_gps.longitude)
    # if no orientation found (standstill)
    if gps_orientation is None:
        # set orientation from last known segment
        k_nearest.loc[0, 'same_lane_dir'] = last_dir_bool
        # check if road has at least one lane
        if last_dir_bool:
            if k_nearest.loc[0, 'sameDirection'] < 1:
                k_nearest.loc[0, 'sameDirection'] = 1
        else:
            if k_nearest.loc[0, 'againstDirection'] < 1:
                k_nearest.loc[0, 'againstDirection'] = 1
        return k_nearest.head(1)

    for idx in k_nearest.index:
        diff_orientation = get_orient_diff(gps_orientation, k_nearest.loc[idx, 'orientation'])
        # apply offset dependant on last orientation
        if last_dir_bool:
            diff_orientation -= 20
        else:
            diff_orientation += 20
        if diff_orientation < 90:
            # same direction
            k_nearest.loc[idx, 'same_lane_dir'] = True
            # set emission higher according to difference in orientation (max 0.3 up)
            k_nearest.loc[idx, 'emissionProb'] += 2 / (7 + diff_orientation)
        else:
            k_nearest.loc[idx, 'same_lane_dir'] = False
            # set emission higher according to difference in orientation (max 0.3 up)
            k_nearest.loc[idx, 'emissionProb'] += 2 / (7 + diff_orientation)

        # check if lanes are available in this direction
        if not check_lane_available(k_nearest, idx):
            k_nearest.loc[idx, 'emissionProb'] -= 1
            # overwrite map with 1 lane
            if k_nearest.loc[idx, 'same_lane_dir']:
                k_nearest.loc[idx, 'sameDirection'] = 1
                k_nearest.loc[idx, 'fromWidth_cm'] = k_nearest.loc[idx, 'toWidth_cm']
            else:
                k_nearest.loc[idx, 'againstDirection'] = 1
                k_nearest.loc[idx, 'toWidth_cm'] = k_nearest.loc[idx, 'fromWidth_cm']

            # optional plot
            if plot_errors:
                fig = plt.figure()
                plt.plot([start_gps.longitude, end_gps.longitude], [start_gps.latitude, end_gps.latitude])
                plt.scatter(start_gps.longitude, start_gps.latitude, marker='o', s=50)
                plt.scatter(end_gps.longitude, end_gps.latitude, marker='^', s=50)
                for i_temp in k_nearest.index:
                    plt.scatter(k_nearest.loc[i_temp, 'longitude'], k_nearest.loc[i_temp, 'latitude'])
                mplleaflet.show(fig=fig)

            # print warning if most probable value is in wrong direction
            if k_nearest.loc[0, 'emissionProb'] < 0:
                # print("Info: Matched opposite orientation on street with only one direction.")
                pass

    # set highest probability to first element with index 0
    if len(k_nearest) > 1:
        k_nearest = k_nearest.sort_values(by='emissionProb', ascending=False)
        k_nearest = k_nearest.reset_index(drop=True)
    # return most probable road and orientation
    return k_nearest


def check_lane_available(k_nearest: pd.DataFrame, idx: int) -> bool:
    same_side = k_nearest.loc[idx, 'same_lane_dir']
    if same_side:
        n_lanes = k_nearest.loc[idx, 'sameDirection']
    else:
        n_lanes = k_nearest.loc[idx, 'againstDirection']
    if n_lanes < 1:
        # wrong street side matched or corrupt map data
        return False
    else:
        # found lane on predicted side
        return True
