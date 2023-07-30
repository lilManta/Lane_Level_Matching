####################################################
## Main script, executes all files in right order
## and setup needed parameters
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 04.2020
####################################################

from baysis.BAYSIS_API import baysis_api
from utils.ask_parameter import ask_parameter
from utils.get_excel_file import get_excel_file
from baysis.plot_bayis_csv import plot_baysis_csv
from digital_map.offset_map import apply_offset
from digital_map.change_driveway_names import change_driveway_names
from markov_model.simple_map_match import simple_map_match
from lane.lane_prob_marg import determine_marg_lane_prob
from lane.combine_prob import combine_gnss_marg_prob
import lane.lane_prob_gnss as lpg
import to_pandas.ToPandas_BAYSIS as tpb
import lane.Map_BAYSISCenterRoads as mbc
import utils.ConvertCoordinates as convert
import digital_map.connect_map as cm
import ground_truth.ground_truth_utils as gtu
import utils.short_utils as usu
import config.paths as paths
import config.config as config
import pickle

#########################
# Get dataset from BAYSIS
#########################
print("[1] Get dataset from BAYSIS? [y or n]")
# q = ask_parameter(parameter='get_baysis_data', param_type=['y', 'n'])
q = 'n'
print('BAYSIS API not available in this extract. Skipping.')

if q == 'y':
    bbox_flag = True  # no need for max_features if true
    max_features = 0
    # max_features = ask_parameter('BAYSIS max_features [int]', 'int')

    for coord in config.type_coordinate_baysis:
        # coord = None    #use default coordinates

        for idx_name in range(len(config.type_name_baysis_bestand)):
            bestand = config.type_name_baysis_bestand[idx_name]
            service = config.type_service_baysis[0]  # bestand
            baysis_api(bestand, service, coord, max_features, bbox_flag, paths, config)

        for idx_name in range(len(config.type_name_baysis_netz)):
            net = config.type_name_baysis_netz[idx_name]
            service = config.type_service_baysis[1]  # netz
            baysis_api(net, service, coord, max_features, bbox_flag, paths, config)


print("[2] Convert BAYSIS dataset to pandas dataframe? [y or n]")
# q = ask_parameter(parameter='convert_to_pd', param_type=['y', 'n'])
print('BAYSIS API not available in this extract. Skipping.')

# use Baysis format for 2020 (True) or 2019 (False)
new_format = True
from_wgs = True
if q == 'y':
    # tpb
    # lane_width to pd
    xml_path = paths.baysis_fahrbahnbreite_gps_file
    tpb.baysis_laneWidth_to_pandas(xml_path, new_format=new_format)

    # lane to pd [gps]
    xml_path = paths.baysis_stock_gps_file
    tpb.baysis_lanes_to_pandas(xml_path, gps=True, new_format=new_format)

    # ways to pd
    xml_path = paths.baysis_net_file  # sectionnumber missing about half the time!
    tpb.baysis_ways_to_pandas(xml_path, printFailures=0, new_format=True)

    print("Convert to Krueger format")
    # Lanes
    pa = paths.baysis_pd_la_gps_file
    with open(pa, 'rb') as package:
        df_Lanes = pickle.load(package)
    for idx in df_Lanes.index:
        lon = df_Lanes.loc[idx, 'coordinatesEast']
        lat = df_Lanes.loc[idx, 'coordinatesNorth']
        df_converted = convert.convert_krueger_to_wgs84(east=lon, north=lat, from_epsg=4258, to_epsg=31468)
        np_converted = df_converted.to_numpy(dtype=float)
        df_Lanes.at[idx, 'coordinatesEast'] = np_converted[:, 0]
        df_Lanes.at[idx, 'coordinatesNorth'] = np_converted[:, 1]
    pa = paths.baysis_pd_la_krueger_file
    df_Lanes.to_pickle(pa)
    del df_Lanes

    # combine lanes gps and krueger
    tpb.baysis_lanes_combine_Kruger_GPS()


print("[3] Preprocess map data? [y or n]")
# q = ask_parameter(parameter='preprocess road data', param_type=['y', 'n'])
print('BAYSIS API not available in this extract. Skipping.')
if q == 'y':
    with open(paths.baysis_pd_lw_file, 'rb') as package:
        LaneWidth = pickle.load(package)
    with open(paths.baysis_pd_la_gps_file, 'rb') as package:
        Lanes = pickle.load(package)
    with open(paths.baysis_pd_wa_file, 'rb') as package:
        Ways = pickle.load(package)

    # use only lane 5
    Lanes = Lanes[Lanes.laneNumber == 5]
    # length unit from m to km
    Lanes.roadLength = Lanes.roadLength / 1000
    Ways.sectionLength = Ways.sectionLength / 1000

    df_mapdata_pre = cm.connectGridNumber(Ways, Lanes)
    df_mapdata = cm.connectGridNumberWidth(df_mapdata_pre, LaneWidth)

    # Correct offset from BAYSIS WGS84 coordinates (not needed for new baysis xml format)
    # df_mapdata = apply_offset(df_mapdata)

    # Apply roadName change for driveways and exits
    df_mapdata = change_driveway_names(df_mapdata)

    print("Finished concatenation of road data sets. Preparing dataset for next part...")

    # # if plot of map dataset wanted
    plot_baysis_csv(df_mapdata, new_format=True, output_sorted_map=False)

    Roads = df_mapdata.rename(columns={'latitude': 'centerRoadNorth',
                                       'longitude': 'centerRoadEast'})
    pickle_save_path = paths.created_pd_center_krueger_file
    mbc.addKruegerCoordinatesToCenterRoads(Roads, pickle_save_path, flattened=True)


##################################
# Run calculations for measurement
##################################

# Show information about map match
with open(paths.excel_file_cursor) as f:
    measurement_name = f.read()
# remove .txt from name
measurement_name = measurement_name[:-5]
print(f"Running Measurement: {measurement_name}")

show_plots = False
print("[4] Process Map Matching? [y or n]")
q = ask_parameter(parameter='process map match', param_type=['y', 'n'])
if q == 'y':
    with open(paths.measurements_path_gps, 'rb') as package:
        GPS = pickle.load(package)

    with open(paths.created_pd_center_krueger_file, 'rb') as package:
        centerRoads_withKR = pickle.load(package)

    # Get sample time
    Sample = 1
    # Sample = ask_parameter(parameter='Sample time for map match', param_type='uint')

    GPS = GPS.reset_index()

    # Determine Map Match
    filter_size = 6
    pathTaken = simple_map_match(GPS, centerRoads_withKR, Sample, filter_size,
                                 savePathTaken=True, plotMapOfMatch=show_plots)


print("[5] Process lane probability? [y or n]")
q = ask_parameter(parameter='process lane probability', param_type=['y', 'n'])
if q == 'y':
    ################################
    # Process GNSS probability model
    ################################
    with open(paths.matching_pd_path_taken_file, 'rb') as package:
        pathTaken = pickle.load(package)

    # update lane information, improve number of lanes, calculate emission probability gnss
    pathTaken = lpg.determine_lane_information(pathTaken)

    # show results
    usu.show_street_usage(pathTaken)

    print('Roads in measurement: ')
    print(pathTaken.roadName.value_counts())

    #############################
    # Process MARG lane maneuver model
    #############################
    with open(paths.measurements_path_acc, 'rb') as package:
        acc = pickle.load(package)

    with open(paths.measurements_path_gps, 'rb') as package:
        gps = pickle.load(package)

    threshold = 20
    speed = usu.get_speed_in_acc_format(gps, acc)
    marg_prob = determine_marg_lane_prob(acc, speed, threshold)

    ####################################
    # combine gnss with marg probability
    ####################################

    # factors in order: [factor_variance, factor_marg, factor_empty_marg]
    factors = [0.05, 0.5, 0.4]
    pathTaken = combine_gnss_marg_prob(pathTaken, marg_prob, gps, factors)

    ###################
    # update dataframes
    ###################
    if True:   # activate for saving results in pathTaken.pkl (overwrite)
        pathTaken.to_pickle(paths.matching_pd_path_taken_file)
        print("Updated pathTaken for comparison checks")


print("[6] Compare detected lane changes with ground truth?\n" +
      "Warning: ground truth has to be manually written in excel sheet. [y or n]")
q = ask_parameter(parameter='compare lane changes', param_type=['y', 'n'])
if q == 'y':
    if True:  # use saved values
        show_plots = False
        with open(paths.matching_pd_path_taken_file, 'rb') as package:
            pathTaken = pickle.load(package)

    excel_file = get_excel_file(paths.excel_file_cursor)
    # import ground_truth table
    ground_truth = gtu.import_excel_file(excel_file, only_lc=True)

    # Get lane change events only from combined gnss and marg probability
    LaneChanges_df, _ = gtu.get_lane_change_events_from_pathTaken(pathTaken)
    # LaneChanges_df, _ = gtu.get_lane_change_events(CombinedProb_df)
    if LaneChanges_df.empty:
        print("Could not find any lane changes.")
        # exit()
    else:
        # Create excel compare sheet for validation purpose
        gtu.write_compare_to_excel(LaneChanges_df, ground_truth, excel_file)

        # Check for correct assigned lane changes
        # assign maximum difference in seconds between program and ground_truth times
        max_time = 3  # [s] if available in ground_truth, the duration of the lane change is added
        found_true, gt_cut_len = gtu.get_correct_lane_changes(ground_truth, pathTaken, max_time)

        # Show result for lane change detection
        gt_cut_len = 0  # if ground_truth completely available
        gtu.print_results(found_true, ground_truth, LaneChanges_df, gt_cut_len)

    # Further analysis if pathTaken is complete
    if 'currentLane' in pathTaken.columns:
        # Check for correct assigned lane numbers
        ground_truth_full = gtu.import_excel_file(excel_file, only_lc=False)
        lane_diff, correct_lane_course = gtu.get_correct_lane_course(ground_truth_full, pathTaken, buffer=max_time)

        # Check for correct assigned lane numbers for multi lanes only
        gtu.get_correct_multi_lane_course(ground_truth_full, pathTaken, buffer=max_time)

        ###############################
        # Plot gps and lane information
        ###############################
        if show_plots:
            with open(paths.measurements_path_gps, 'rb') as package:
                gps = pickle.load(package)
            import ground_truth.gt_plot as gtp

            # Plot wrong matched lanes
            save_plot_path = paths.map_plot_path + 'plot_wrong_lanes.html'
            gtp.plot_wrong_lanes(pathTaken, gps, lane_diff, save_plot_path)

            # Plot correct lane course
            save_plot_path = paths.map_plot_path + 'plot_lanes_ground_truth_vs_current_lane.html'
            pred_lane_course = pathTaken['currentLane'].tolist()
            gtp.plot_lane_course(pathTaken, gps, correct_lane_course,
                                 pred_lane_course, save_plot_path)

            # Plot lane course vs. maximum lanes available
            save_plot_path = paths.map_plot_path + 'plot_lanes_ground_truth_vs_map_lanes.html'
            pred_lane_course = pathTaken['numberOfLanes'].tolist()
            gtp.plot_lane_course(pathTaken, gps, correct_lane_course,
                                 pred_lane_course, save_plot_path)


print("Finished main code.")
