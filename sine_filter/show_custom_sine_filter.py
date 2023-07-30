import numpy as np
# import pickle
import matplotlib.pyplot as plt

import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from lane.lane_prob_marg import createLCFilter

####################################################
## This script is part of the optimization and
## shows the custom generated lc filter script
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 09.2020
####################################################


def show_filter(filter_name):
    # load filter from numpy save object
    custom_filter = np.load(filter_name)

    # create x axis
    len_filter = len(custom_filter)
    stop = len_filter/2
    stop = round(stop/100, 1)
    x = np.linspace(-stop, stop, len_filter)

    # for lc_type in [1, -1]:
    for lc_type in [1]:
        fontsize = 25
        plt.rcParams.update({'font.size': fontsize})

        # create custom filter plot
        # fig, ax = plt.subplots()
        plt.figure(figsize=(11, 7.6))
        custom_filter *= lc_type
        plt.plot(x, custom_filter, 'k', lw=3, label='Custom curve')

        # create normal sine filter plot
        sine_filter = create_sine_filter(len_filter)
        sine_filter *= lc_type
        plt.plot(x, sine_filter, '--', color='#0065BD', lw=2, label='Sine curve')

        # label axis
        plt.xlabel('Maneuver time in s')
        plt.ylabel('Normed amplitude')
        plt.title('Custom sine filter for lane changes', fontsize=fontsize)
        if lc_type == -1:
            plt.title(f"Angepasster Sinus Filter für Spurwechsel nach links [{stop*2}s]")
        else:
            plt.title(f"Angepasster Sinus Filter für Spurwechsel nach rechts [{stop*2}s]")
        plt.legend()

        # show plot
        plt.grid(axis='y')
        plt.show()


def create_sine_filter(maneuver_time):
    # create standard sine wave for lane change right
    sample_time = 1
    filter_amp = 0.5
    _, filter_right = createLCFilter(sample_time=sample_time, maneuver_time=maneuver_time,
                                     filterAmplitude=filter_amp, laneChange=1)

    return filter_right


if __name__ == '__main__':
    file_path = 'custom_sine_filter_40.npy'

    show_filter(file_path)
