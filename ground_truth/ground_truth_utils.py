import pandas as pd
import numpy as np
import datetime


####################################################
## This script contains several small functions for
## the ground_truth algorithm used in the main function
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 07.2020
####################################################


def import_excel_file(file_path: str, only_lc=True) -> pd.DataFrame:
    """ This function imports the ground_truth data from an excel sheet.
    The sheet has to be created manually by watching the video!

    :param only_lc:     if True returns only the lane changes, if False also road changes
    :param file_path:   complete file path to excel sheet
    :return:            ground truth table, transformed as pandas dataframe
    """

    # ground_truth = pd.read_excel(file_path, columns=['Type', 'Type ID'])
    ground_truth = pd.read_excel(file_path)
    # drop header rows
    ground_truth = ground_truth.drop([0, 1, 2, 3, 4, 5, 6])
    # drop empty rows if available (e.g. time, time_from_start and type must not be empty)
    ground_truth = ground_truth.dropna(thresh=3)

    # drop rows with type_ID 5x,6x or 7x
    if only_lc:
        for idx_gt in ground_truth.index:
            if not ground_truth.at[idx_gt, 'type'].startswith('lane change '):
                ground_truth = ground_truth.drop(idx_gt)

    ground_truth = ground_truth.reset_index(drop=True)
    # drop "time_from_start" ("time" is still inside!)
    ground_truth = ground_truth.drop(columns='time_from_start')

    return ground_truth


def get_lane_change_events(CombinedProb_df) -> pd.DataFrame:
    # Reduce program result dataframe to lane changes (L/R)
    LaneChanges_df = CombinedProb_df[CombinedProb_df.currentLaneInfo != '']
    LaneChanges_df = LaneChanges_df[LaneChanges_df.currentLaneInfo != 'Road Change']

    LaneChanges_df = LaneChanges_df.reset_index(drop=True)

    if len(LaneChanges_df.index) == 0:
        print("Could not find any lane changes in measurement!")

    time_lane_changes = LaneChanges_df.time_round_S
    # convert to time object
    for t_lc in time_lane_changes.index:
        time_lane_changes[t_lc] = time_lane_changes[t_lc].time()

    return LaneChanges_df, time_lane_changes


def get_lane_change_events_from_pathTaken(pathTaken):
    lane_changes = pd.DataFrame(columns=pathTaken.columns)
    for idx in pathTaken.index:
        if pathTaken.loc[idx, 'laneChangeDirection'] != '':
            lane_changes = lane_changes.append(pathTaken.loc[idx], ignore_index=True)

    if lane_changes.empty:
        return lane_changes, None

    # convert timestamp to datetime.time format
    lane_changes = lane_changes.rename(columns={'timestamp': 'time'})
    lane_changes['time_round_S'] = lane_changes.time.dt.round('S')
    lane_changes['time_round_S'] = lane_changes['time_round_S'].apply(lambda x: x.time())

    time_lane_changes = lane_changes.time_round_S

    return lane_changes, time_lane_changes


def calc_time_diff(t1: datetime.time, t2: datetime.time, absolute=True) -> float:
    # calculate absolute time difference between two pandas.datetime.time objects
    sec1 = (t1.hour * 60 + t1.minute) * 60 + t1.second
    sec2 = (t2.hour * 60 + t2.minute) * 60 + t2.second
    diff = sec1 - sec2  # return diff in seconds
    if absolute:
        diff = abs(diff)
    return diff


def write_compare_to_excel(LaneChanges_df, ground_truth_df, excel_file_path: str) -> None:
    # Create df with both information (program and manual found lane changes)
    lc_df_prep = pd.DataFrame(LaneChanges_df.currentLaneInfo)
    lc_df_prep = lc_df_prep.join(LaneChanges_df.laneChangeDirection)
    lc_df_prep = lc_df_prep.join(LaneChanges_df.time_round_S)
    lc_df_prep = lc_df_prep.rename(columns={'currentLaneInfo': 'type',
                                            'laneChangeDirection': 'type_ID',
                                            'time_round_S': 'time'})

    excel_df = ground_truth_df.append(lc_df_prep, ignore_index=True)
    excel_df = excel_df.sort_values('time')
    excel_df.to_excel(excel_file_path[:-5] + '_test.xlsx')


def get_gt_lane_direction(gt_type):
    if gt_type == 'lane change left':
        gt_lane_direction = 'L'
    elif gt_type == 'lane change right':
        gt_lane_direction = 'R'
    else:
        print("Did not understand lane change type: {}".format(gt_type))
        gt_lane_direction = ''
    return gt_lane_direction


def get_correct_lane_changes(ground_truth_df, pathTaken, max_time: float) -> (int, int):
    print_wrong_direction = False

    # Return number of correct lane changes
    found_true = 0
    lc_time_used = []
    # Get calculated lane changes
    LaneChanges_df, time_lane_changes = get_lane_change_events_from_pathTaken(pathTaken)
    # Remove last elements from laneChanges_df if there is no ground_truth available
    gt_cut = 30  # Set video confirmation cut after last known lane change
    if time_lane_changes is None:
        return 0, 0
    for lc_time in time_lane_changes:
        diff = calc_time_diff(lc_time, list(ground_truth_df.time)[-1], absolute=False)
        if diff > gt_cut:
            # do not compare time since its out of ground truth bounds
            lc_time_used.append(lc_time)
    gt_cut_len = len(lc_time_used)

    for idx_gt in ground_truth_df.index:
        gt_time = ground_truth_df.at[idx_gt, 'time']
        gt_lane_direction = get_gt_lane_direction(ground_truth_df.loc[idx_gt, 'type'])
        # Add duration to max_time
        max_time_gt = max_time
        if ground_truth_df.at[idx_gt, 'duration_in_sec'] is not None:
            max_time_gt += ground_truth_df.at[idx_gt, 'duration_in_sec']
        for idx_lc, lc_time in enumerate(time_lane_changes):
            if lc_time not in lc_time_used:
                # search for detected lane change
                if calc_time_diff(gt_time, lc_time) < max_time_gt:
                    if LaneChanges_df.loc[idx_lc, 'laneChangeDirection'] == gt_lane_direction:
                        found_true += 1
                        lc_time_used.append(lc_time)
                        break
                    else:
                        if print_wrong_direction:
                            print("Found lane change <{}> in wrong direction.".format(idx_lc))
    return found_true, gt_cut_len


def print_results(found_true: int, ground_truth, LaneChanges_df, gt_cut_len: int) -> None:
    idx_0 = found_true
    idx_1 = len(ground_truth.index)
    print("Found {} out of {} true lane changes ({}%)".format(idx_0, idx_1, round(100 * idx_0 / idx_1, 1)))

    idx_0 = found_true
    idx_1 = len(LaneChanges_df) - gt_cut_len
    if idx_1 == 0:
        print('No lane changes found in window')
        return None
    print("True {} out of {} predicted lane changes ({}%)".format(idx_0, idx_1, round(100 * idx_0 / idx_1, 1)))

    idx_0 = round((idx_1 - found_true) * 100 / idx_1, 2)
    print("False-positive rate: {}%".format(idx_0))


def calc_results(found_true: int, ground_truth, LaneChanges_df, gt_cut_len: int) -> (float, float):
    idx_0 = found_true
    idx_1 = len(ground_truth.index)
    true_lc = round(100 * idx_0 / idx_1, 1)

    idx_0 = found_true
    idx_1 = len(LaneChanges_df) - gt_cut_len
    if idx_1 <= 0:
        pred_lc = 0
    else:
        pred_lc = round(100 * idx_0 / idx_1, 1)

    return true_lc, pred_lc


def get_correct_lane_course(ground_truth, pathTaken, return_lane_comparison=False, buffer=2) -> list:
    """ This function prints the accuracy of the lane prediction compared
        to ground_truth from excel sheet

    :param ground_truth: excel file loaded as pandas dataframe WITH all rows
    :param pathTaken: full road course dataframe WITH fixed sample rate
    :param return_lane_comparison: flag if return percentage or lane_course
    :return: list of diff in lane prediction
             if element == 0:   correct lane
             if element > 0:    real/ground_truth lane is to the right
             if element < 0:    real/ground_truth lane is to the left

    """
    found_true = 0
    lane_diff_list = []
    correct_lane_course = []
    current_lane = 1
    gt_time_element = 0

    for idx in pathTaken.index:
        # get current road time
        current_time = pathTaken.loc[idx, 'timestamp'].time()

        # check for lane change
        while current_time >= ground_truth.loc[gt_time_element, 'time']:
            # update lane number
            current_lane = ground_truth.loc[gt_time_element, 'lane_number']
            gt_time_element += 1
            if gt_time_element == len(ground_truth):
                # finish
                current_lane = 1
                break

        if min(pathTaken.loc[idx - buffer:idx + buffer + 1, 'currentLane'] - current_lane) == 0:
            found_true += 1
        # save result in dictionary
        lane_diff_list.append(min(pathTaken.loc[idx - buffer:idx + buffer + 1, 'currentLane'] - current_lane))
        correct_lane_course.append(current_lane)

    # Print results
    s1 = found_true
    s2 = len(pathTaken)
    s3 = round(100 * s1 / s2, 1)

    if return_lane_comparison:
        return s3
    else:
        print(f"Lane number comparison: {s1}/{s2} true ({s3}% of the time)")

        return lane_diff_list, correct_lane_course


def get_correct_multi_lane_course(ground_truth, pathTaken, return_lane_comparison=False, buffer=1):
    found_true = 0
    found_true_second = 0
    current_lane = 1
    gt_time_element = 0

    lanes_gt = []
    prob_true = []
    prob_false = []

    # create ground truth database
    for idx in pathTaken.index:
        # get current road time
        current_time = pathTaken.loc[idx, 'timestamp'].time()

        # check for lane change
        while current_time >= ground_truth.loc[gt_time_element, 'time']:
            # update lane number
            current_lane = ground_truth.loc[gt_time_element, 'lane_number']
            gt_time_element += 1
            if gt_time_element == len(ground_truth):
                # finish
                current_lane = 1
                break
        lanes_gt.append(current_lane)

    # setup temporary lane buffer with lane_1 entries
    lanes_pathTaken = pathTaken.loc[:, 'currentLane'].tolist()
    max_lanes_pathTaken = pathTaken.loc[:, 'numberOfLanes'].tolist()
    combinedProb = pathTaken.loc[:, 'combinedProb'].tolist()
    if len(lanes_gt) != len(lanes_pathTaken):
        print(f"Error: lanes_gt {len(lanes_gt)} vs lanes_pathTaken {len(lanes_pathTaken)}")

    # iterate through lane buffer to check correct lane
    for idx in range(len(lanes_pathTaken)):
        # check for multi lane
        if max_lanes_pathTaken[idx] > 1:
            # setup lane buffer
            if idx - buffer > 0:
                if idx + buffer + 1 >= len(lanes_gt):
                    # ending segment
                    lane_buffer = lanes_gt[idx - buffer:]
                    # print(f"Found end segment at idx {idx} of {len(lanes_gt)}")
                else:
                    # normal segment
                    lane_buffer = lanes_gt[idx - buffer:idx + buffer + 1]
            else:
                # starting segment
                # print(f"Found start segment at idx {idx}")
                lane_buffer = lanes_gt[0:idx + buffer + 1]

            # check if current lane is inside buffer
            if lanes_pathTaken[idx] in lane_buffer:
                found_true += 1
                found_true_second += 1
                prob_true.append(combinedProb[idx])
            # check one lane error
            elif min(abs(np.asarray(lane_buffer) - lanes_pathTaken[idx])) < 1.5:
                found_true_second += 1
            else:
                prob_false.append(combinedProb[idx])

    # check if baysis max roads is exceeded
    baysis_errors = 0
    for idx in range(len(lanes_pathTaken)):
        if max_lanes_pathTaken[idx] < lanes_gt[idx]:
            baysis_errors += 1

    # Print results
    s1 = found_true
    s2 = sum(pathTaken.numberOfLanes > 1)
    s3 = round(100 * s1 / s2, 1)
    s4 = baysis_errors
    s5 = round(100 * s4 / len(pathTaken), 1)
    s6 = round(100 * found_true_second / s2, 1)

    if return_lane_comparison:
        return s3

    print(f"Multi-lane number comparison: {s1}/{s2} true ({s3}% of the time)")
    print(f"Multi-lane rate with 1 lane errors allowed: {s6}%")
    print(f"Unpossible lanes found in map data on course: {s4}/{len(pathTaken)} ({s5}%)")

    # calculate probability for true and false matches
    prob_true = np.asarray(prob_true)
    prob_false = np.asarray(prob_false)
    # calculate lane difference
    lane_diff = np.asarray(lanes_gt) - np.asarray(lanes_pathTaken)

    print(f"Multi-lane lane difference: lanes_ground_truth-lanes_pathTaken\n"
          f" mean:   {np.mean(lane_diff):.3f}\n"
          f" median: {np.median(lane_diff):.3f}\n"
          f" std:    {np.std(lane_diff):.3f}\n"
          # f" var:    {np.var(lane_diff):.3f}\n"
          f" min:    {np.min(lane_diff)}\n"
          f" max:    {np.max(lane_diff)}")
    # print(f"Multi-lane probability comparison:\n  true matches | false matches\n"
    #       f" mean:   {np.mean(prob_true):.3f} | {np.mean(prob_false):.3f}\n"
    #       f" median: {np.median(prob_true):.3f} | {np.median(prob_false):.3f}\n"
    #       f" std:    {np.std(prob_true):.3f} | {np.std(prob_false):.3f}\n"
    #       f" var:    {np.var(prob_true):.3f} | {np.var(prob_false):.3f}\n"
    #       f" 25quan: {np.quantile(prob_true, 0.25):.3f} | {np.quantile(prob_false, 0.25):.3f}\n"
    #       f" 75quan: {np.quantile(prob_true, 0.75):.3f} | {np.quantile(prob_false, 0.95):.3f}\n"
    #       f" max:    {np.max(prob_true):.3f} | {np.max(prob_false):.3f}")

    return None
