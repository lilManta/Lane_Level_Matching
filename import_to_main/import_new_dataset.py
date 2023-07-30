import pandas as pd
# import pickle
# import numpy as np
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import paths
from datetime import datetime
from progressbar import ProgressBar

pbar = ProgressBar()

####################################################
## This script loads all data needed from a csv
## into the format needed for the main function
##--> Needs to be run for every new dataset
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 09.2020
####################################################


def replace_measurement(folder_path: str, ground_truth_name: str) -> None:
    if len(os.listdir(folder_path)) != 1:
        print(f"Only one file allowed in folder: {folder_path}")

    for file in os.listdir(paths.measurements_path):
        if file.endswith(".pkl"):
            os.remove(paths.measurements_path + file)

    for file in os.listdir(folder_path):
        pd_all = pd.read_csv(folder_path + file)

    pd_all = pd_all.rename(columns={'Timestamp': 'time'})
    pd_all = str2datetime(pd_all)

    pd_acc = pd_all[['time', 'acc_x', 'acc_y', 'acc_z']].copy()
    pd_gyro = pd_all[['time', 'gyro_x', 'gyro_y', 'gyro_z']].copy()
    pd_gps = pd_all[['time', 'latitude', 'longitude', 'speed']].copy()
    pd_gps = down_sample(pd_gps, 100)

    pd_acc = pd_acc.rename(columns={'acc_x': 'X', 'acc_y': 'Y', 'acc_z': 'Z'})
    pd_acc.to_pickle(paths.measurements_path_acc)
    print('Successfully overwritten Acceleration measurement. [1/4]')

    pd_gps.to_pickle(paths.measurements_path_gps)
    print('Successfully overwritten GPS measurement. [2/4]')

    pd_gyro.to_pickle(paths.measurements_path_gyro)
    print('Successfully overwritten Gyro measurement. [3/4]')

    with open(paths.excel_file_cursor, 'w') as txt_file:
        txt_file.write(ground_truth_name + '.xlsx')
    print('Successfully overwritten current measurement cursor file [4/4]')


def down_sample(df: pd.DataFrame, step: int) -> pd.DataFrame:
    # take every <step> sample
    df = df[::step]
    # reset index
    df = df.reset_index(drop=True)

    return df


def str2datetime(dataframe):
    """ Converts the time column of the dataframe from string to pandas datetime format

    :param dataframe: pd.DataFrame with column 'time' as string format
    :return: pd.DataFrame with column 'time' as pandas datetime64 format
    """
    print("Convert time format to pandas datetime. May take a while")
    date_temp = []

    for index in dataframe.index:
    # for index in pbar(dataframe.index):
        # get time string
        string = dataframe.time.values[index].rstrip()
        # append datetime object to temporary list
        date_temp.append(datetime.strptime(string, '%Y-%m-%d %H:%M:%S.%f'))

    # save temporary list to dataframe
    dataframe.time = date_temp

    return dataframe


if __name__ == '__main__':
    # folder name where to find the tracks as files Acceleration_ and GPS_ (make sure to use "/")
    folder_path = paths.measurements_path + 'Audi_A3/'

    ground_truth_name = '200424-1339-1425'
    # ground_truth_name = '200424-1433-1537'
    # ground_truth_name = '200427-1042-1113'
    # ground_truth_name = '200427-1113-1144'
    # ground_truth_name = '200427-1145-1228'
    # ground_truth_name = '200427-1228-1310'

    folder_path += ground_truth_name
    folder_path += '/GS4_1/sectionMethod/'
    # folder_path += '/GP2/sectionMethod/'
    # measurement path can be adjusted in config.paths
    replace_measurement(folder_path, ground_truth_name)

    print('\nFinished overwriting measurements. Continue with main.')
