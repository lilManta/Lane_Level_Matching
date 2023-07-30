import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplleaflet

####################################################
## This script contains several small plot functions for
## the ground_truth algorithm used in the main function
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 07.2020
####################################################


def plot_wrong_lanes(pathTaken, gps, lane_diff: list, save_path: str) -> None:
    # find wrong matched lane numbers
    path_idx_list = [i for i in range(len(lane_diff)) if lane_diff[i] != 0]
    pathTaken = pathTaken.reset_index(drop=True)

    fig = plt.figure()
    lon = gps.loc[:, 'longitude']
    lat = gps.loc[:, 'latitude']
    plt.plot(lon, lat, linewidth=3)
    # iterate over wrong matched lanes
    for idx in path_idx_list:
        # get gps coordinates
        gps_idx = pathTaken.loc[idx, 'GPSindex']
        lon = gps.loc[gps_idx, 'longitude']
        lat = gps.loc[gps_idx, 'latitude']
        # label = lane_diff[idx]
        plt.scatter(lon, lat, c='r', marker='D')

    mplleaflet.show(fig=fig, path=save_path)


def plot_lane_course(pathTaken, gps, correct_lane_course: list,
                     pred_lane_course: list, save_path: str) -> None:
    # Warning: only works for continuous measurement (all gps points are aligned)
    # plot all matched gps points and color by lane number

    # get gps coordinate points
    gps_indices = pathTaken['GPSindex']
    gps_reduced = gps.loc[gps_indices]
    gps_reduced = gps_reduced.reset_index(drop=True)

    # open plot figure
    fig = plt.figure()

    # initialize start values
    color_lane = ['empty', 'b', 'r', 'y', 'k', 'g']
    color_lane_label = '<lane number colors> 1: blue, 2: red, 3: yellow, 4: black 5: green'
    count = 1
    second_offset = 0.001

    # check if predicted data is given
    if len(pred_lane_course) > 1:
        course = [correct_lane_course, pred_lane_course]
    else:
        course = [correct_lane_course]

    # switch between ground_truth and predicted lane data (first element is correct_lane_course)
    for lane_course in course:

        if count == 2:
            # offset gps points for second iteration (predicted_lane_course only)
            gps_reduced['longitude'] = gps_reduced['longitude'] + second_offset
            gps_reduced['latitude'] = gps_reduced['latitude'] - second_offset/2

        # start iteration over lanes
        last_lane = lane_course[0]
        lon = []
        lat = []
        for idx, current_lane in enumerate(lane_course):
            if current_lane != last_lane:
                # create new line with lane color
                if len(lon) > 0:
                    plt.plot(lon, lat, c=color_lane[last_lane], linewidth=4)
                lon = []
                lat = []
                last_lane = current_lane

            # add coordinates for plot
            lon.append(gps_reduced.loc[idx, 'longitude'])
            lat.append(gps_reduced.loc[idx, 'latitude'])

        # plot last course
        if len(lon) > 0:
            plt.plot(lon, lat, c=color_lane[last_lane], linewidth=4)

        # set count for next iteration
        count += 1

    # show plot in mplleaflet
    mplleaflet.show(fig=fig, path=save_path)

    # save legend as txt file (not supported in mplleaflet)
    txt_path = save_path[:-5] + '_legend.txt'
    with open(txt_path, 'w') as txt_file:
        txt_file.write(color_lane_label)





