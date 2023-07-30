import numpy as np
import pandas as pd
from utils.get_df_index_to_timestamp import get_df_index_to_timestamp

####################################################
## This script is for fusing the calculated lane
## probabilities from gnss and marg
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################


def combine_gnss_marg_prob(pathTaken, marg_prob, gps, factors: list):
    """ This function combines the results from lane_prob_gnss.py and
        lane_prob_marg.py and returns the pathTaken dataframe with all information

    :param pathTaken: pd.Dataframe from lane_prob_gnss function
    :param marg_prob: pd.Dataframe from lane_prob_marg function
    :param gps:       pd.Dataframe from measurement gps (needed for timestamps)
    :param factors:   list of tuning factors in order
                      [factor_variance, factor_marg, factor_empty_marg]
    :return: updated pathTaken dataframe with combined lane probability
    """
    # # Tuning parameters
    # factor_marg = 1
    # factor_gnss = 1
    # thresh_combined = -2
    # timesteps_gnss_variance = 4

    # add timestamps to pathTaken dataframe
    for idx in pathTaken.GPSindex:
        pathTaken.loc[idx, 'timestamp'] = gps.loc[idx, 'time']

    # update pathTaken columns order for visual purpose only
    pathTaken = pathTaken[['timestamp', 'roadName', 'GPSindex', 'currentLane', 'numberOfLanes',
                           'distance', 'emissionProb', 'same_lane_dir', 'fromNode', 'toNode',
                           'longitude', 'latitude', 'matchedEast_Kruger', 'matchedNorth_Kruger']]

    # reset currentLane to lane 1 (most right)
    pathTaken.loc[:, 'currentLane'] = 1
    # add column laneChangeDirection and combinedProb
    pathTaken.insert(loc=5, column='laneChangeDirection', value='')
    pathTaken.insert(loc=6, column='combinedProb', value='')

    # set number of street parts to compare (should be uneven)
    k = 7
    # calculate gnss variance
    pathTaken = calc_gnss_course_prob(pathTaken, k)

    # set marg probability inside pathTaken
    pathTaken = add_marg_to_pathTaken(pathTaken, marg_prob)

    # combine marg and gnss lane calculation
    pathTaken = combine_pathTaken(pathTaken, factors)

    return pathTaken


def combine_pathTaken(pathTaken, factors=[]):
    if len(factors) == 3:
        factor_variance = factors[0]
        factor_marg = factors[1]
        factor_empty_marg = factors[2]
    else:
        factor_variance = 1.5
        factor_marg = 0.7
        factor_empty_marg = 0.7

    # iterate through street segments
    last_road = pathTaken.loc[pathTaken.index[0], 'roadName']

    for idx in pathTaken.index:

        # check if street is changing
        if last_road != pathTaken.loc[idx, 'roadName']:
            # reset road variables and set to lane 1
            last_road = pathTaken.loc[idx, 'roadName']
            # pathTaken.loc[idx:, 'currentLane'] = 1
            # skip index
            continue

        # check if map match has multiple lanes
        if pathTaken.loc[idx, 'numberOfLanes'] > 1:
            # add variance to gps probability
            gps_prob = np.asarray(pathTaken.at[idx, 'emissionProb']).copy()
            gps_prob *= 3   # set emphasis on gps course
            gps_prob += factor_variance * np.asarray(pathTaken.loc[idx, 'varianceProb'])
            # update dataframe
            pathTaken.at[idx, 'emissionProb'] = gps_prob

            if factor_variance > 1:
                gps_prob /= (1 + factor_variance)/2

            # get current best lane guess
            prob = np.max(gps_prob)
            lane = np.argmax(gps_prob) + 1
            # update dataframe
            pathTaken.loc[idx, 'combinedProb'] = prob
            pathTaken.loc[idx, 'currentLane'] = lane

    # fill na values in combinedProb with max definition
    pathTaken.combinedProb = pd.to_numeric(pathTaken.combinedProb).fillna(2)

    # apply marg prob
    pathTaken = apply_marg_prob(pathTaken, factor_marg, factor_empty_marg)

    # apply temporal filter
    pathTaken = apply_temporal_filter(pathTaken)

    # add lane change information to dataframe
    pathTaken = add_lc_info(pathTaken)

    return pathTaken


def add_lc_info(pathTaken):
    # start in most right lane
    last_lane = 1

    # reset lane information
    pathTaken.loc[:, 'laneChangeDirection'] = ''
    pathTaken.loc[:, 'currentLaneInfo'] = ''

    for idx in pathTaken.index:
        if pathTaken.loc[idx, 'currentLane'] != last_lane:
            # get lane change direction
            if pathTaken.loc[idx, 'currentLane'] > last_lane:
                pathTaken.loc[idx, 'laneChangeDirection'] = 'L'
                pathTaken.loc[idx, 'currentLaneInfo'] = 'LCL'
            else:
                pathTaken.loc[idx, 'laneChangeDirection'] = 'R'
                pathTaken.loc[idx, 'currentLaneInfo'] = 'LCR'

            # update last lane
            last_lane = pathTaken.loc[idx, 'currentLane']

    return pathTaken


def apply_marg_prob(pathTaken, factor_marg: float, factor_empty_marg: float):
    # add lane changes from marg to lane course from gps
    lane_before = 1
    for idx in pathTaken.index:
        # skip last index
        if idx == pathTaken.index[-1]:
            break

        lc_dir = None
        # check for lc in marg
        if pathTaken.loc[idx, 'margDir'] == 'R':
            if lane_before > 1:
                lc_dir = 'R'
        elif pathTaken.loc[idx, 'margDir'] == 'L':
            if lane_before < pathTaken.loc[idx, 'numberOfLanes']:
                lc_dir = 'L'

        # add to gps course
        # if lc_dir is not None:
        # calculate lane array
        marg_ar = [0]*5
        if lc_dir == 'R':
            marg_ar[lane_before-2] = 1
            marg_ar = factor_marg * pathTaken.loc[idx, 'margProb'] * np.asarray(marg_ar)
        elif lc_dir == 'L':
            marg_ar[lane_before] = 1
            marg_ar = factor_marg * pathTaken.loc[idx, 'margProb'] * np.asarray(marg_ar)
        else:   # None
            # set emphasis on current lane (only apply to current index)
            marg_ar[lane_before-1] = factor_empty_marg
            marg_ar = np.asarray(marg_ar)

        # add to dataframe
        pathTaken.at[idx, 'emissionProb'] += marg_ar
        pathTaken.loc[idx, 'currentLane'] = np.argmax(pathTaken.at[idx, 'emissionProb']) + 1
        if lc_dir is not None:
            # apply reduced factor to tail
            factor_tail = 0.4
            idx_1 = pathTaken.index[pathTaken.index.get_loc(idx)-1]
            idx_3 = pathTaken.index[pathTaken.index.get_loc(idx)+1]
            # save in dataframe (tail)
            pathTaken.at[idx_1, 'emissionProb'] += factor_tail * marg_ar
            pathTaken.at[idx_3, 'emissionProb'] += factor_tail * marg_ar
            # update current lane (tail)
            new_lane = np.argmax(pathTaken.at[idx_1, 'emissionProb']) + 1
            if new_lane <= pathTaken.loc[idx_1, 'numberOfLanes']:
                pathTaken.loc[idx_1, 'currentLane'] = new_lane
            new_lane = np.argmax(pathTaken.at[idx_3, 'emissionProb']) + 1
            if new_lane <= pathTaken.loc[idx_3, 'numberOfLanes']:
                pathTaken.loc[idx_3, 'currentLane'] = new_lane

        # update last lane
        lane_before = pathTaken.loc[idx, 'currentLane']

    return pathTaken


def apply_temporal_filter(pathTaken):
    # TODO: real temporal filter
    for idx in pathTaken.index:
        # check if finished
        if idx == pathTaken.index[-2]:
            break

        # get next indices
        idx_2 = pathTaken.index[pathTaken.index.get_loc(idx)+1]
        idx_3 = pathTaken.index[pathTaken.index.get_loc(idx)+2]

        # check if first and last idx have same lane number
        if pathTaken.loc[idx, 'currentLane'] == pathTaken.loc[idx_3, 'currentLane']:
            # check if lane in middle idx is different
            if pathTaken.loc[idx_2, 'currentLane'] != pathTaken.loc[idx, 'currentLane']:
                # change middle element to same as around
                pathTaken.loc[idx_2, 'currentLane'] = pathTaken.loc[idx, 'currentLane']

    return pathTaken


def add_marg_to_pathTaken(pathTaken, marg_prob):
    pathTaken['margProb'] = ''
    pathTaken['margDir'] = ''

    # get timestamps from accelerometer probability
    for idx in marg_prob.index:
        time_marg = marg_prob.loc[idx, 'timestamp']

        # get pathTaken index for this time
        pathTaken_idx = get_df_index_to_timestamp(pathTaken, 'timestamp', time_marg)
        if pathTaken_idx is not None:
            pathTaken.loc[pathTaken_idx, 'margDir'] = marg_prob.loc[idx, 'direction']
            pathTaken.loc[pathTaken_idx, 'margProb'] = round(marg_prob.loc[idx, 'probability'], 2)

    return pathTaken


def calc_gnss_course_prob(pathTaken, k: int):
    """ Calculate the variance from course shift in lane level and add to dataframe

    :param pathTaken: full dataframe
    :param k: number of neighbors around idx to calculate temporal shift/variance
    :return: pathTaken with variance
    """
    # initialize
    prob_list = []
    idx_list = []
    last_road = None
    pathTaken.loc[:, 'varianceProb'] = ''

    for idx in pathTaken.index:
        prob_ar = pathTaken.loc[idx, 'emissionProb']
        prob_list.insert(0, prob_ar)
        idx_list.insert(0, idx)

        if pathTaken.loc[idx, 'roadName'] != last_road:
            # road change found - reset variance
            last_road = pathTaken.loc[idx, 'roadName']
            prob_list = [prob_list[0]]
            idx_list = [idx_list[0]]

        if len(prob_list) >= k:
            # hold up to k elements
            if len(prob_list) > k:
                prob_list.pop(-1)
                idx_list.pop(-1)
            # calc variance
            variance_ar = calc_variance(prob_list, k)

        else:
            # not enough samples for shift
            variance_ar = [0]*5
        # add to dataframe

        pathTaken.at[idx, 'varianceProb'] = variance_ar

    return pathTaken


def calc_variance(prob_list: list, k: int) -> list:
    if len(np.unique(prob_list)) <= 2:
        # only one lane and rest 0
        variance = [0]*5
        return variance

    # set current lane to end and oldest lane to beginning
    prob_list.reverse()

    half = int(k/2)
    # get older half
    first_half = prob_list[:half]
    # get newer half
    second_half = prob_list[half:]
    # cut off middle element in newer half if needed
    if len(second_half) > len(first_half):
        second_half.pop(0)

    first_half_mean = np.mean(first_half, axis=0)
    second_half_mean = np.mean(second_half, axis=0)

    variance = second_half_mean - first_half_mean
    variance = np.round(variance, 2)
    return variance
