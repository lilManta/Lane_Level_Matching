import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import mplleaflet
from filters.Filter_BidirectionalButterworth import BBPF
import filters.custom_filters as cf
from progressbar import ProgressBar
pbar = ProgressBar()

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import paths

####################################################
## This script is for calculating the lane prob
## from marg (acc) sensor data
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################


def determine_marg_lane_prob(acc, speed: pd.Series, threshold: float) -> pd.DataFrame:
    """Determine the lane changes by comparing the accelerometer X-axis to a sine wave

    :param acc: pandas dataframe with acceleration measurement 'X'
    :param speed: speed in same sample rate as acc data
    :param threshold: similarity needed between sine filter and actual measurement data
    :return: dataframe with: timestamp for each detected lane change
                             direction for each detected lane change ('L' or 'R')
    """
    # # Define presets
    # threshold = 80.0
    sample_time = 0.01
    # maneuver_time = 5
    # filter_amp = 0.4
    # acc_sample_time = (acc.loc[len(acc)-1, 'time'] - acc.loc[0, 'time']).delta / 1e9 / (len(acc) - 1)
    # reductionFactor = sample_time / acc_sample_time
    threshold_delta_LR = (0, 0)

    lc_probability = np.array([])
    lc_timestamp = np.array([])
    lc_direction = np.array([])

    # Create the Sine Lane change filter
    if False:
        maneuver_time, filter_amp = 4, 0.45
        _, filt = createLCFilter(sample_time=sample_time, maneuver_time=maneuver_time,
                                        filterAmplitude=filter_amp, laneChange=1)
    else:
        filt = np.load(paths.custom_sine_filter)
        # calculate maneuver time (filter length * sample rate)
        maneuver_time = len(filt)*sample_time
    maneuver_samples = len(filt)


    # Filter the Data with Butterworth filter (forwards and backwards)
    low_pass_acc = BBPF(acc, signal='X', sampleRate=100, highCut=0.3, order=4)
    # normalize to +- 1
    if sample_time != 0.01:
        sampler = int(sample_time*100)
        if sampler < 1:
            print("Error: positive sample time greater 0.01 needed.")
            exit()
        low_pass_acc = low_pass_acc[::sampler]
    # low_pass_acc = cf.sklearn_normalize(low_pass_acc)
    low_pass_orig = low_pass_acc.copy()
    low_pass_acc = cf.sqrt_scale(low_pass_acc)
    # low_pass_acc = cf.sqrt_scale(low_pass_acc)
    # df = cf.window_normalize(df)
    low_pass_acc = cf.sklearn_scale(low_pass_acc)

    # Determine length of measurement
    num_measurements = len(low_pass_acc)

    # Run through measurement, Determine Segment, Error, if less than threshold add to detected maneuvers
    delayAfterDetection = maneuver_time
    threshold_last = None
    count_probability_check = 0
    for idx in pbar(range(num_measurements - maneuver_samples)):
    # for idx in range(num_measurements - maneuver_samples):
        # calculate dynamic threshold
        cur_speed = speed[idx + int(maneuver_samples/2)]
        dynamic_thresh = threshold + (cur_speed/3)*(threshold/60)

        # check new maneuvers after 0.5s of previous maneuver (avoid multi detection)
        if delayAfterDetection >= 0.5:
            # cut measured segment to compare to sine filter
            # segment = low_pass_acc[idx: idx + maneuver_samples * int(sample_time * 100):int(sample_time * 100)]
            segment = low_pass_acc[idx: idx + maneuver_samples]
            # compare to sine filter for left and right lane change
            epsLCL = determineEps(segment, filt * -1, cur_speed)
            epsLCR = determineEps(segment, filt, cur_speed)

            if (epsLCL < dynamic_thresh + threshold_delta_LR[0]) |\
                    (epsLCR < dynamic_thresh + threshold_delta_LR[1]):
                # save timestamp from middle of maneuver
                lc_timestamp = np.append(lc_timestamp, acc.loc[int(idx + maneuver_samples/2), 'time'])
                delayAfterDetection = 0
                if epsLCL - threshold_delta_LR[0] < epsLCR - threshold_delta_LR[1]:
                    # lane change left found
                    lc_direction = np.append(lc_direction, 'L')
                    threshold_last = epsLCL
                else:
                    # lane change right found
                    lc_direction = np.append(lc_direction, 'R')
                    threshold_last = epsLCR
        else:
            if threshold_last is None:
                if count_probability_check <= 0:
                    delayAfterDetection = delayAfterDetection + sample_time
                else:
                    count_probability_check -= 1
            if threshold_last is not None:
                count_probability_check += 1
                # Start evaluation again for specific direction to check if the probability is increasing
                segment = low_pass_acc[idx: idx + maneuver_samples]
                if lc_direction[-1] == 'L':
                    eps = determineEps(segment, filt * -1, cur_speed)
                else:
                    eps = determineEps(segment, filt, cur_speed)

                if eps > threshold_last:
                    # probability is decreasing -> take last threshold as probability
                    lc_probability = np.append(lc_probability, (dynamic_thresh/2) / (1+threshold_last))
                    # stop searching for a higher probability
                    threshold_last = None
                else:
                    threshold_last = eps

    # merge results in single dataframe
    acc_prob = pd.DataFrame([lc_timestamp, lc_direction, lc_probability]).T
    acc_prob.columns = ['timestamp', 'direction', 'probability']
    return acc_prob


def createLCFilter(sample_time=0.01, maneuver_time=6.0, filterAmplitude=1.0, laneChange=1):
    """This function creates the sine filter array needed to compare with real acc data

    :param sample_time:     sample time for simulation in seconds
    :param maneuver_time:   needed time for one lane change in seconds
    :param filterAmplitude: factor for amplitude of sine wave
    :param laneChange:      1 for lane change right, -1 for lane change left
    :return:t [np.ndarray]: time vector from 0 to maneuver_time
            f [np.ndarray]: sine wave vector
    """
    t = np.linspace(0, maneuver_time, int(maneuver_time / sample_time), dtype=np.float)
    f = laneChange * filterAmplitude * np.sin(t / maneuver_time * 2 * np.pi)

    return t, f


def determineEps(acc_data: np.ndarray, filter: np.ndarray, speed: float) -> float:
    """ This function sums the absolute difference between filter and data and returns this as EPS """
    # check if moving
    if speed == 0:
        # stopped vehicle
        eps = 99999
        return eps

    # center window
    acc_data = cf.center_data(acc_data)

    # calculate sum of difference
    eps = np.sum(np.absolute(filter - acc_data))

    return eps