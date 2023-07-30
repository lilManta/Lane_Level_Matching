import pandas as pd

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import paths

####################################################
## This script is a container for small functions
## used in the main scripts
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 06.2020
####################################################


def get_speed_in_acc_format(gps: pd.DataFrame, acc: pd.DataFrame) -> pd.Series:

    gps = gps.set_index('time')
    gps = gps['speed']
    speed = gps.reindex(acc.time).interpolate()

    return speed


def show_street_usage(pathTaken: pd.DataFrame) -> None:
    # initialize
    pathTaken = pathTaken.reset_index(drop=True)
    count_list_street = []
    street_list = [pathTaken.loc[0, 'roadName']]
    count = 0

    # sort street usage by street name
    ##################################
    for idx in pathTaken.index:
        current_street = pathTaken.loc[idx, 'roadName']
        count += 1

        if current_street != street_list[-1]:
            # add to list and reset count
            street_list.append(current_street)
            count_list_street.append(count)
            count = 0

    # append last count to list
    count_list_street.append(count)

    # sort street usage by lane number
    ##################################
    nlanes_series = pathTaken.numberOfLanes
    lane_1 = sum(nlanes_series == 1)
    lane_2 = sum(nlanes_series == 2)
    lane_3 = sum(nlanes_series == 3)
    lane_4 = sum(nlanes_series == 4)
    lane_5 = sum(nlanes_series == 5)
    count_list_lane = [lane_1, lane_2, lane_3, lane_4, lane_5]

    # save results as text file
    ###########################
    save_file = paths.digital_map_path + 'street_usage.txt'
    with open(save_file, 'w') as f:
        # street list
        for idx in range(len(street_list)):
            f.writelines(f"Street: {street_list[idx]}, seconds: {count_list_street[idx]}\n")
        # lane list
        for idx in range(len(count_list_lane)):
            perc = round(100*count_list_lane[idx] / len(pathTaken), 1)
            f.writelines(f"\nLane {idx + 1}, seconds: {count_list_lane[idx]}, percentage: {perc}")

    print(f"Saved street usage as {save_file}")
