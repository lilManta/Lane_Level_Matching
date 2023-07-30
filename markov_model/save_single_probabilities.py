import pandas as pd
import numpy as np

####################################################
## This script is used in the main function to skip
## usage of gps XOR acc signal
## It allows to analyse one sensor only in part 6
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 09.2020
####################################################


def save_gnss_prob(pathTaken, gps, save_file_name: str) -> None:
    """ This function saves the result of the gnss probability
        for comparing with ground truth in part 6

    :param pathTaken: gnss probability result dataframe
    :param gps: gps dataframe for timestamps
    :param save_file_name: pickle file path
    :return: None
    """
    if 'laneChangeDirection' in pathTaken.columns:
        print("GNSS probability is already saved. Continue")
        return None
    pathTaken.insert(loc=9, column='laneChangeDirection', value='')
    pathTaken.insert(loc=10, column='currentLaneInfo', value='')
    last_lane = pathTaken.loc[pathTaken.index[0], 'currentLane']

    for idx in pathTaken.index:
        if pathTaken.loc[idx, 'currentLane'] > last_lane:
            # lane change left
            pathTaken.loc[idx, 'laneChangeDirection'] = 'L'
            pathTaken.loc[idx, 'currentLaneInfo'] = 'LCL'
            last_lane = pathTaken.loc[idx, 'currentLane']

        elif pathTaken.loc[idx, 'currentLane'] < last_lane:
            # lane change right
            pathTaken.loc[idx, 'laneChangeDirection'] = 'R'
            pathTaken.loc[idx, 'currentLaneInfo'] = 'LCR'
            last_lane = pathTaken.loc[idx, 'currentLane']

    # add timestamps
    pathTaken.loc[:, 'time'] = gps.time[pathTaken.index]

    # save gnssProb
    pathTaken.to_pickle(save_file_name)


def save_marg_prob(marg_prob, save_file_name):
    """ This function saves the result of the marg probability
        for comparing with ground truth in part 6

    :param marg_prob: marg probability result dataframe
    :param save_file_name: pickle file path
    :return: None
    """
    # rename columns
    marg_prob = marg_prob.rename(columns={'timestamp': 'time', 'direction': 'laneChangeDirection'})
    # add lane info
    for idx in marg_prob.index:
        marg_prob.loc[idx, 'currentLaneInfo'] = 'LC' + marg_prob.loc[idx, 'laneChangeDirection']

    if save_file_name == 'return':
        return marg_prob
    # save gnssProb
    marg_prob.to_pickle(save_file_name)


def create_empty_marg(pathTaken, gps) -> pd.DataFrame:
    """ This function returns an empty marg dataframe to apply in combine_gnss_marg_prob()

    :param pathTaken: DataFrame with gnss data only
    :return: empty marg dataframe
    """
    if 'timestamp' not in pathTaken.columns:
        for idx in pathTaken.index:
            pathTaken.loc[idx, 'timestamp'] = gps.loc[idx, 'time']

    marg = pd.DataFrame(pathTaken.loc[:, 'timestamp'])

    # add probabilities same as result in determine_marg_lane_prob()
    marg.insert(loc=1, column='direction', value='')
    marg.insert(loc=2, column='probability', value=0.0)
    switch_flag = True
    for idx in marg.index:
        if switch_flag:
            marg.loc[idx, 'direction'] = 'R'
        else:
            marg.loc[idx, 'direction'] = 'L'
        switch_flag = not switch_flag

    print("###\nWarning: overwritten marg probability with empty dataframe for debugging!\n###")
    return marg

