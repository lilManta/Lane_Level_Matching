##########################################################
## This script contains all file paths
## Workspace is setup automatically
##########################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
##########################################################
## Date: 05.2020
##########################################################
## Important!
## Only use "/" (not "\")
## Always end with "/", e.g. C:/user/folder/
##########################################################

import os

# workspace and data paths
########################## (dir_path is automatically adjusted for any PC)
dir_path = os.path.abspath(os.getcwd())
if not os.path.isfile(dir_path + '/main.py'):
    dir_path = os.path.dirname(dir_path)
    if not os.path.isfile(dir_path + '/main.py'):
        dir_path = os.path.dirname(dir_path)
    if not os.path.isfile(dir_path + '/main.py'):
        print("Could not find root directory. Start with project folder set to main folder "
              "or edit workspace manually in 'config/paths.py'. Exit.")
        exit()
dir_path += '/'

# path for measurement files, can also be on a different disk
data_path = dir_path + 'data/'
##########################

# folder structure code
#######################
baysis_path = dir_path + 'baysis/'
filter_path = dir_path + 'filter/'
utils_path = dir_path + 'utils/'
raw_plot_path = dir_path + 'sine_filer/rawData_plots/'
digital_map_path = dir_path + 'digital_map/'

# folder structure data
#######################
baysis_data_path = data_path + 'baysis_data/'
measurements_path = data_path + 'measurements/'
map_plot_path = data_path + 'map_plots/'
pandas_baysis_path = data_path + 'Dataframes/Baysis/'
created_path = data_path + 'Dataframes/CreatedMap/'
created_edit_path = created_path + 'Edits/'
matching_path = data_path + 'Dataframes/MapMatching/'
ground_truth_path = data_path + 'ground_truth_lane_changes/'

# files
#######
#baysis_api
baysis_fahrbahnbreite_gps_file = baysis_data_path + 'STRBESTAND_WFS_Fahrbahnbreiten_WGS84.xml'

baysis_hatches_file = baysis_data_path + 'STRNETZ_WFS_Stationierung_Hatches100_WGS84.xml'

baysis_node_file = baysis_data_path + 'STRNETZ_WFS_Netzknoten_WGS84.xml'

baysis_net_file = baysis_data_path + 'STRNETZ_WFS_Strassennetz_WGS84.xml'

baysis_stock_gps_file = baysis_data_path + 'STRBESTAND_WFS_Fahrstreifen_WGS84.xml'

baysis_stock_krueger_file = baysis_data_path + 'STRBESTAND_WFS_Fahrstreifen_Krueger.xml'

#dataframes/baysis
baysis_pd_lw_file = pandas_baysis_path + 'LaneWidth_WGS84.pkl'
baysis_pd_ha_file = pandas_baysis_path + 'Hatches_Krueger.pkl'
baysis_pd_no_file = pandas_baysis_path + 'Nodes_Krueger.pkl'
baysis_pd_la_file = pandas_baysis_path + 'Lanes_both.pkl'
baysis_pd_la_gps_file = pandas_baysis_path + 'Lanes_GPS.pkl'
baysis_pd_la_krueger_file = pandas_baysis_path + 'Lanes_Krueger.pkl'
baysis_pd_wa_file = pandas_baysis_path + 'Ways_Krueger.pkl'
#dataframes/created_map
created_pd_mapdata_file = created_path + 'mapdata.pkl'
created_pd_center_file = created_path + 'centerRoads.pkl'
created_pd_center_krueger_file = created_path + 'centerRoads_withKrueger.pkl'
created_pd_center_krueger_sorted_file = created_path + 'centerRoads_withKrueger_sorted.pkl'
created_pd_imp_lane_width_file = created_path + 'importantLaneWidths.pkl'
#dataframes/map_matching
matching_pd_path_taken_file = matching_path + 'pathTaken.pkl'
matching_pd_path_taken_new_file = matching_path + 'pathTaken_new.pkl'
# ground truth
excel_file_cursor = ground_truth_path + 'current_measurement.txt'
# new_baysis_map
map_data_csv = baysis_data_path + 'mapdata_final_1.3.csv'
# measurements
measurements_path_acc = measurements_path + 'Acc_datetime.pkl'
measurements_path_gps = measurements_path + 'Gps_datetime.pkl'
measurements_path_gyro = measurements_path + 'Gyro_datetime.pkl'
measurements_path_ori = measurements_path + 'Orient_datetime.pkl'
custom_sine_filter = dir_path + 'sine_filter/custom_sine_filter_40.npy'
