import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import mplleaflet
# from utils.DetermineDistance import geoDistancePythagoras

####################################################
## This script is for calculating the lane prob
## from gnss (gps) sensor data
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################


def determine_lane_information(pathTaken):
    """ This function calculates the lane width and updates the lane information in pathTaken dataframe

    :param pathTaken: matched road dataset
    :return: pathTaken dataframe updated with current lane information
    """

    if 'fromWidth_cm' not in pathTaken.columns:
        print('GNSS probability calculation already done. If you want to rerun start part 4: map matching first')
        return pathTaken

    pathTaken['emissionProb'] = pathTaken['emissionProb'].astype('object')

    # initialize current lane (all to right lane)
    pathTaken.loc[:, 'currentLane'] = 1

    # fill nan values with next value
    pathTaken.fromWidth_cm = pathTaken.fromWidth_cm.bfill()
    pathTaken.toWidth_cm = pathTaken.toWidth_cm.bfill()

    for idx in pathTaken.index:
        # get lane widths
        if pathTaken.loc[idx, 'same_lane_dir']:
            lane_width = pathTaken.loc[idx, 'fromWidth_cm']
        else:
            lane_width = pathTaken.loc[idx, 'toWidth_cm']

        number_lanes = pathTaken.loc[idx, 'numberOfLanes']

        # check baysis number of lanes
        calc_lanes, warn_lanes = correct_baysis_lanes(lane_width, number_lanes)
        warn_lanes = False
        if warn_lanes:
            print(f"Warning calc_lanes: {calc_lanes} vs baysis_lanes: {number_lanes} with lane_wdith: {lane_width} "
                  f"gps coordinates {pathTaken.loc[idx, 'latitude']},"
                  f"{pathTaken.loc[idx, 'longitude']}")

        if calc_lanes != 0:
            pathTaken.loc[idx, 'numberOfLanes'] = calc_lanes
            number_lanes = calc_lanes

        # skip if road has only one lane
        if pathTaken.loc[idx, 'numberOfLanes'] == 1:
            # pathTaken.at[idx, 'emissionProb'] = [pathTaken.loc[idx, 'emissionProb']]
            pathTaken.at[idx, 'emissionProb'] = [1, 0, 0, 0, 0]
            continue

        # get lane distance list
        if pathTaken.loc[idx, 'roadName'].startswith('A'):
            highway = True
        else:
            highway = False
        lane_distance = get_splitted_lane_distance(lane_width, number_lanes, highway)

        # get gps distance
        distance = pathTaken.loc[idx, 'distance']

        # calculate emission probability on lane level
        emission = lane_distance - distance
        emissionProb = get_emission_probability(emission)

        if len(emissionProb) == 0 or sum(emissionProb) == 0:
            print(f"Error in lane_prob_gnss: emssisionProb={emissionProb}")
            exit()

        # update pathTaken dataframe
        pathTaken.at[idx, 'emissionProb'] = np.round(emissionProb, 2)
        pathTaken.loc[idx, 'currentLane'] = 1 + np.argmax(emissionProb)

    # drop unnecessary columns
    pathTaken = pathTaken.drop(columns=['orientation', 'fromWidth_cm', 'toWidth_cm'])

    return pathTaken


def get_splitted_lane_distance(lane_width_max: float, number_lanes: int, highway: bool) -> list:
    # simple split width of complete road on lane widths
    lanes_distance = []

    # new_width = 0
    if highway:
        # most left lane has offset from middle
        new_width = 2
    else:
        # most left lane is on the border of the middle line
        new_width = 0.1

    step_size = (lane_width_max - new_width) / number_lanes
    new_width -= step_size / 2

    for i in range(number_lanes):
        new_width += step_size
        lanes_distance.append(new_width)
    # reverse lane width order from 1: most right lane to 5: most left lane
    lanes_distance = lanes_distance[::-1]

    return lanes_distance


def get_emission_probability(emission: np.ndarray, add_end=0) -> np.ndarray:
    emission = abs(emission)

    # get probability by inverse
    emissionProb = 1 / (1 + abs(emission))

    # distribute to range 0-1
    max_emission = max(emissionProb)
    emissionProb = emissionProb / max_emission

    # add all other lanes with 0 (not possible)
    if len(emissionProb) < 5:
        diff = 5 - len(emissionProb)
        emissionProb = list(emissionProb)
        emissionProb.extend([add_end]*diff)

    return emissionProb


def correct_baysis_lanes(width: float, n_lanes: int) -> (int, bool):
    """ This script redefines the number of lanes if data from baysis is corrupt

    :param n_lanes: number of all lanes together
    :param width: width of all lanes together
    :return: number of lanes if different else 0 + warning flag
    """
    # define standard lane width in meter
    normal_lane_width = 3.8

    # Check lane minimum of 3 meters each
    if width < 6:
        # too narrow for multiple lanes. Skipping
        return 0, False

    # calculate possible lanes by width
    calc_lanes = int(round(width / normal_lane_width, 0))

    # compare calculated lanes to baysis lanes
    if calc_lanes == n_lanes:
        return 0, False

    if calc_lanes > n_lanes:

        # check if out of bound (max 5 lanes)
        if calc_lanes > 5:
            calc_lanes = 5
            return calc_lanes, True

        return calc_lanes, False

    else:
        # print("lower lanes predicted")
        return calc_lanes, True

