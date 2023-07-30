import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplleaflet
import time
from progressbar import ProgressBar
pbar = ProgressBar()

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from markov_model.simple_orient_match import simple_orient_match
import config.paths as paths
from utils import ConvertCoordinates as convert


####################################################
## This script is map-matching the gps points on the
## street. It has been written as an substitute for
## the HiddenMarkovModel
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 06.2020
####################################################


def simple_map_match(gps, centerRoads, SampleTime=1, gps_filter_size=5,
                     savePathTaken=True, plotMapOfMatch=True):
    start_time = time.time()
    gps = preprocessGPSdata(gps, SampleTime)
    gps = gps_smooth_filter(gps, box_size=gps_filter_size, show_plot=False)
    fail_list = []
    k = 8   # find up to k neighbor roads (1 gps point -> k different roads)
    timestamps_roadName = 4  # compare current road name with last <x> matched road points
    timestamps_orientation = 4  # gps coordinates forward and backward to feed to orientation matching algo
    last_dir_bool = True    # initialize same or against road flag (updated if data available)
    road_matches_df = pd.DataFrame()
    road_matches_df_all = {}
    apply_offset_flag = False
    offset = [0, 0]
    # for idx in gps.index:
    for idx in pbar(gps.index):
        k_nearest = get_k_nearest(gps.loc[idx, 'east'] + offset[0],
                                  gps.loc[idx, 'north'] + offset[1],
                                  centerRoads, k)
        # TODO: find out direction and map to same-/against-Direction
        if k_nearest is None:
            # print("Could not match GPS point with index: {}" .format(idx))
            fail_list.append(idx)
        else:
            # add GPS index
            k_nearest.insert(0, 'GPSindex', idx)

            # apply emission probability logic
            k_nearest = apply_emission_logic(k_nearest, road_matches_df.tail(timestamps_roadName))

            # get start and end coordinates for orientation logic
            start_idx = idx - timestamps_orientation
            if start_idx < gps.index[0]:
                start_idx = gps.index[0]
            end_idx = idx + timestamps_orientation
            if end_idx > gps.index[-1]:
                end_idx = gps.index[-1]
            start_gps = gps.loc[start_idx, ['longitude', 'latitude']]
            end_gps = gps.loc[end_idx, ['longitude', 'latitude']]

            # apply orientation logic and return single row
            k_nearest_copy = simple_orient_match(k_nearest, start_gps, end_gps, last_dir_bool)
            k_nearest = k_nearest_copy.head(1)
            last_dir_bool = k_nearest.loc[0, 'same_lane_dir']

            # change dataframe format for part 5
            k_nearest = apply_format(k_nearest)

            # add to result dataframe
            if road_matches_df.empty:
                road_matches_df = k_nearest
            else:
                road_matches_df = pd.concat([road_matches_df, k_nearest], ignore_index=True)

            if len(k_nearest_copy) > 1:
                # append full k_nearest to dictionary with key idx
                road_matches_df_all[idx] = k_nearest_copy

            if apply_offset_flag:
                offset = update_offset(road_matches_df, gps.loc[idx, 'east'],
                                       gps.loc[idx, 'north'], offset)

    if road_matches_df.empty:
        print("Error. Could not match any GPS coordinates. Exit")
        exit()

    # iterate through match and search for single street segments
    #############################################################
    street_array = []
    # set overwrite up to x matches if single segment found
    overwrite_up_to = 6
    street_memory = overwrite_up_to + 2
    for idx in road_matches_df.index:
        # add current street name to first element of list
        street_array.insert(0, road_matches_df.loc[idx, 'roadName'])

        # if memory length is full
        if len(street_array) > street_memory:
            # remove oldest street name
            street_array.pop(-1)

            # start name comparison
            if len(np.unique(street_array)) > 1:
                if street_array[0] == street_array[-1]:
                    # delete single segments and try to add street from dict
                    road_matches_df = update_street_match(road_matches_df, idx,
                                                          street_array, road_matches_df_all)
                    # start new search
                    street_array = [street_array[0]]


    # edit dataframe format for part 5
    ##################################
    road_matches_df = road_matches_df.set_index('GPSindex', drop=False)
    road_matches_df.sort_index(inplace=True)
    road_matches_df.numberOfLanes = road_matches_df.numberOfLanes.bfill()
    road_matches_df.numberOfLanes = road_matches_df.numberOfLanes.astype('int')

    # print time needed
    print('Time needed for simple map match: {0:.1f}s '.format(time.time() - start_time))
    # print(f'Failed GPS indices: {fail_list}')
    print(f'Failed GPS indices: {len(fail_list)}')

    if savePathTaken:
        print('Saving Dataframe: ' + paths.matching_pd_path_taken_new_file)
        road_matches_df.to_pickle(paths.matching_pd_path_taken_file)
    if plotMapOfMatch:
        print('Plotting matched road dataset')
        plot_simple_match(road_matches_df, gps)

    return road_matches_df


def gps_smooth_filter(gps, box_size, show_plot=False):
    east = gps.east.to_numpy()
    north = gps.north.to_numpy()

    box = np.ones(box_size) / box_size
    # filter east values
    east_sm = np.convolve(east, box, mode='same')
    # replace first and last values with original value due to filter
    east_sm[:box_size] = east[:box_size]
    east_sm[-box_size:] = east[-box_size:]
    # filter north values
    north_sm = np.convolve(north, box, mode='same')
    # replace first and last values with original value due to filter
    north_sm[:box_size] = north[:box_size]
    north_sm[-box_size:] = north[-box_size:]

    if show_plot:
        fontsize = 25
        plt.rcParams.update({'font.size': fontsize})
        plt.figure(figsize=(11, 7.6))
        plt.plot(east, north, '--', color='#0065BD',  lw=2, marker='o', markersize=8, label='GPS signal')
        plt.plot(east_sm, north_sm, '-k', lw=3, label='GPS with filter')
        plt.legend(loc='lower right')
        plt.xlabel('East in m')
        plt.ylabel('North in m')
        plt.title('GPS data filter', fontsize=fontsize)
        plt.show()

    gps.east = east_sm
    gps.north = north_sm

    return gps


def update_street_match(road_matches_df: pd.DataFrame, idx: int,
                        street_array: list, road_matches_df_all: dict):
    added_flag = False
    correct_street = street_array[0]
    for street_idx in range(1, len(street_array)-1):
        if street_array[street_idx] != correct_street:
            gps_index = road_matches_df.loc[idx-street_idx, 'GPSindex']
            # delete match from dataframe
            road_matches_df = road_matches_df.drop(index=idx-street_idx)
            # try to find street in dictionary
            df_temp = road_matches_df_all.get(gps_index)
            if df_temp is not None:
                for temp_idx in df_temp.index:
                    if df_temp.loc[temp_idx, 'roadName'] == correct_street:
                        # add street to dataframe
                        road_matches_df.loc[idx-street_idx] = df_temp.loc[temp_idx]
                        added_flag = True

    if added_flag:
        road_matches_df.sort_index(inplace=True)

    return road_matches_df


def update_offset(k_nearest, east: float, north: float, offset: list) -> list:
    # initialize factor to new offset vs old offset
    k = 0.15
    # check k_nearest must have more than one row
    if len(k_nearest) > 1:
        ind = k_nearest.index[-2:].tolist()

        if k_nearest.loc[ind[1], 'numberOfLanes'] == 1:
            diff_e = k_nearest.loc[ind[1], 'matchedEast_Kruger'] - east
            diff_n = k_nearest.loc[ind[1], 'matchedNorth_Kruger'] - north
            offset = apply_update_offset(offset, diff_e, diff_n, k)

    return offset


def apply_update_offset(offset: list, diff_east: float, diff_north: float, k: float) -> list:
    # set limit for gps offset in meters
    limit = 5
    # update offset
    if abs((offset[0] + diff_east*k) / (1 + k)) < limit:
        offset[0] = (offset[0] + diff_east*k) / (1 + k)
    if abs((offset[1] + diff_north*k) / (1 + k)) < limit:
        offset[1] = (offset[1] + diff_north*k) / (1 + k)

    return offset


def get_k_nearest(east: float, north, centerRoads: pd.DataFrame, k: int) -> pd.DataFrame:
    # set maximum distance between points to declare as neighbor
    max_distance = 45       # in meter

    # create absolute difference in centerRoads
    diff_east = centerRoads.centerRoadEastKrueger.to_numpy() - east
    diff_north = centerRoads.centerRoadNorthKrueger.to_numpy() - north
    # Simplify difference east/north from float64 to float32
    diff_east = diff_east.astype('float32')
    diff_north = diff_north.astype('float32')
    centerRoads['diff_abs'] = get_diff_abs(diff_east, diff_north)

    # drop by max value
    centerRoads = centerRoads[centerRoads.diff_abs < max_distance]
    if centerRoads.empty:
        return None

    # sort by absolute difference
    centerRoads = centerRoads.sort_values(by=['diff_abs'])

    # drop by street name double
    k_nearest = centerRoads.loc[get_first_street_indexes(centerRoads, k)]

    # get emission probability
    k_nearest['emissionProb'] = (max_distance - k_nearest['diff_abs']) / max_distance

    for idx in k_nearest.index:
        test = (max_distance - k_nearest.loc[idx, 'diff_abs']) / max_distance

    # create correct output format
    k_neighbors = k_nearest[['roadName', 'centerRoadEast', 'centerRoadNorth', 'centerRoadEastKrueger',
                             'centerRoadNorthKrueger', 'diff_abs', 'emissionProb', 'orientation',
                             'sameDirection', 'againstDirection', 'fromNode', 'toNode', 'fromWidth_cm',
                             'toWidth_cm', 'offset_sameDir_lon', 'offset_sameDir_lat',
                             'offset_againstDir_lon', 'offset_againstDir_lat']]

    k_neighbors = k_neighbors.reset_index(drop=True)
    k_neighbors.rename(columns={'diff_abs': 'distance',  # 'index': 'CenterRoadindex',
                                'centerRoadEast': 'longitude', 'centerRoadNorth': 'latitude',
                                'centerRoadEastKrueger': 'matchedEast_Kruger',
                                'centerRoadNorthKrueger': 'matchedNorth_Kruger'}, inplace=True)

    return k_neighbors


def get_diff_abs(diff_east: np.ndarray, diff_north: np.ndarray) -> np.ndarray:
    # Calculate absolute difference between two pair of coordinates by Pythagorean theorem
    # Exact result only valid for near matching points because of Krueger coordinates!

    diff_abs = np.sqrt(np.square(diff_east) + np.square(diff_north))

    return diff_abs


def get_first_street_indexes(centerRoads: pd.DataFrame, k: int) -> list:
    # Returns the first k indexes of different streets which appear in centerRoads
    index_list = []
    street_name_list = []
    for idx in centerRoads.index:
        if centerRoads.loc[idx, 'roadName'] not in street_name_list:
            index_list.append(idx)
            if len(index_list) >= k:
                return index_list
            # update last street name
            street_name_list.append(centerRoads.loc[idx, 'roadName'])

    return index_list


def apply_emission_logic(k_nearest, road_matches_tail):
    # set down emission probability for each time the road name
    # appeared last n times but not in actual k_nearest
    if len(k_nearest) <= 1:
        return k_nearest
    up_set = 0.08
    if road_matches_tail.empty:
        # no recent roads
        return k_nearest.head(1)
    last_road_names = road_matches_tail.roadName.to_list()
    # extend list with last value (2x) to put more value on current street
    last_road_names.extend([last_road_names[-1]]*2)
    # extend list with pre-last value (1x)
    last_road_names.append(last_road_names[-4])

    for road_name in last_road_names:
        for idx in k_nearest.index:
            if k_nearest.loc[idx, 'roadName'] == road_name:
                # up set emission probability
                k_nearest.loc[idx, 'emissionProb'] = k_nearest.loc[idx, 'emissionProb'] + up_set

    # down set emission probability for oneWay roads
    down_set = 0.05
    for idx in k_nearest.index:
        if 'oneWay' in k_nearest.loc[idx, 'roadName']:
            if k_nearest.loc[idx, 'emissionProb'] > down_set:
                k_nearest.loc[idx, 'emissionProb'] = k_nearest.loc[idx, 'emissionProb'] - down_set
            else:
                k_nearest.loc[idx, 'emissionProb'] = 0

    # up set emission probability for A roads without oneWay
    # (standard range is much higher due to many lanes -> space to middle of street)
    up_set = 0.3
    for idx in k_nearest.index:
        road_name = k_nearest.loc[idx, 'roadName']
        if road_name.startswith('A') and 'oneWay' not in road_name:
            k_nearest.loc[idx, 'emissionProb'] = k_nearest.loc[idx, 'emissionProb'] + up_set

        elif 'A' in road_name and 'oneWay' not in road_name:
            print(f"May found A road: {road_name}")

    return k_nearest


def plot_simple_match(road_matches_df, gps):
    # plots
    fig = plt.figure()
    # plot gps course
    plt.plot(gps.longitude, gps.latitude, 'g', linewidth=12)
    # plot map match
    plt.scatter(road_matches_df.longitude, road_matches_df.latitude, c='k', marker='x', s=3)

    # # scatter found gps coordinates
    # plt.scatter(gps_found.longitude, gps_found.latitude, s=170, c='b', marker='o')
    # # scatter road matches
    # plt.scatter(road_matches_df.longitude, road_matches_df.latitude, s=140, c='r', marker='X')
    mplleaflet.show(fig=fig, path=paths.map_plot_path + 'Traveled_route_map.html')


def apply_format(k_nearest):
    # Reformat lane number for part 5
    if len(k_nearest) != 1:
        print("Error in simple map match. Dataframe k_nearest has not exactly one row.")
        exit()

    # get current number of lanes in correct direction
    if k_nearest.loc[0, 'same_lane_dir']:
        # set number of lanes
        k_nearest = k_nearest.rename(columns={'sameDirection': 'numberOfLanes'})
        # drop unnecessary information
        k_nearest = k_nearest.drop(columns=['againstDirection',
                                            'offset_againstDir_lon', 'offset_againstDir_lat',
                                            'offset_sameDir_lon', 'offset_sameDir_lat'])

    else:
        k_nearest = k_nearest.rename(columns={'againstDirection': 'numberOfLanes'})
        # drop unnecessary information
        k_nearest = k_nearest.drop(columns=['sameDirection',
                                            'offset_againstDir_lon', 'offset_againstDir_lat',
                                            'offset_sameDir_lon', 'offset_sameDir_lat'])

    return k_nearest


def preprocessGPSdata(gps_data, sample=1):
    """
        This function reduces the GPS data by taking sample measurement and
        add the Krueger coordinates
    :param gps_data: Dataframe which is to be reduced (by sample)
    :param sample: every x gps point is taken
    :return: reduced Dataframe
    """
    # sample down if needed
    gps_reduced = gps_data[gps_data.index % sample == 0]

    # add to Krueger coordinates to Dataframe
    # Convert all GPS points to Kruger
    gps_norm = convert.convert_krueger_to_wgs84(east=gps_reduced.longitude, north=gps_reduced.latitude,
                                                from_epsg=4258, to_epsg=31468)
    # change index of converted to concatenate
    gps_norm.index = gps_reduced.index
    # add krueger coordinates to the reduced dataframe
    gps_reduced = pd.concat([gps_reduced, gps_norm], axis=1)

    return gps_reduced
