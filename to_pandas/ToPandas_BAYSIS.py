import xml.etree.ElementTree as ET
import pandas as pd
import os
import numpy as np
import pickle
import config.paths as paths
from utils.pandas_to_numeric import pandas_to_numeric
from utils.ask_parameter import ask_parameter


#####################################################
## This script reads BAYSIS XML files into Pandas DFs
#####################################################
## Author: Frederic Brenner, concept by Christopher Bennett
## Email: frederic.brenner@tum.de
## Email: christopherbennett92@gmail.com
#####################################################
## Date: 05.2020
#####################################################


def find_xml_search_pattern(root, xml_file_name):
    search_pattern = ""
    baysis1 = '{http://www.opengis.net/gml/3.2}member'
    baysis2 = '{http://www.opengis.net/wfs/2.0}member'
    if len(root.findall(baysis1)) > 1:
        search_pattern = baysis1
    elif len(root.findall(baysis2)) > 1:
        search_pattern = baysis2
    else:
        print(f"Could not find content in search pattern for xml: {xml_file_name}")
        exit()
    return search_pattern


def baysis_laneWidth_to_pandas(path_to_file: str, printFailures=0, new_format=True):
    """
        A quick parse of Baysis Ways to Pandas Dataframe
        ... print errors to console with printFailures
        ... Errors saved to Ways_info.txt
    """
    # Local Variables
    failIDs = ''

    # Erzeugung des XML-Baums
    tree = ET.parse(path_to_file)
    # Das Wurzelelement wird gespeichert
    root = tree.getroot()
    # Ein neuer DataFrame mit den notwendigen Spalten wird initialisiert
    df_width = pd.DataFrame(
        columns=['OBJECT ID', 'roadName', 'roadType', 'sectionNumber', 'fromStation', 'toStation',
                 'laneStripe', 'LaneStripeNumber', 'laneStripeTypeKey', 'laneStripeType', 'fromWidth_cm', 'toWidth_cm',
                 'fromNode', 'fromNodeKey', 'toNode', 'toNodeKey', 'coordinatesEast', 'coordinatesNorth'])

    tag_pre = '{https://www.baysis.bayern.de/gis/services/wfs/BAYSIS_Strassenbestand/MapServer/WFSServer}'
    # Erstellen einer Iterationsvariablen
    i = 0
    # For-Schleifen: Innerhalb der For-Schleifen werden die Unterstrukturen des XML-Files ausgelesen. Darin befinden sich die Informationen für jeden Streckenabschnitt der sich innerhalb
    # der betrachteten Bereichs befindet
    search_pattern = find_xml_search_pattern(root, 'laneWidth')
    for child in root.findall(search_pattern):
        for subchild in child:
            # Innerhalb einer Zeile des DataFrames werden die relevanten Informationen für einen definierten Streckenabschnitt gespeichert
            df_width.loc[i, 'OBJECT ID'] = subchild.find(tag_pre + 'OBJECTID').text
            df_width.loc[i, 'roadName'] = subchild.find(tag_pre + 'Straßenname_').text
            df_width.loc[i, 'roadType'] = subchild.find(tag_pre + 'Straßenklasse_').text
            df_width.loc[i, 'sectionNumber'] = subchild.find(tag_pre + 'Abschnittsnummer_').text
            df_width.loc[i, 'fromStation'] = float(subchild.find(tag_pre + 'Von-Station_').text)
            df_width.loc[i, 'toStation'] = float(subchild.find(tag_pre + 'Bis-Station_').text)
            df_width.loc[i, 'laneStripe'] = subchild.find(tag_pre + 'Streifen_').text
            df_width.loc[i, 'LaneStripeNumber'] = subchild.find(tag_pre + 'Streifennummer_').text
            df_width.loc[i, 'laneStripeTypeKey'] = subchild.find(tag_pre + 'Streifenart__Schlüssel__').text
            df_width.loc[i, 'laneStripeType'] = subchild.find(tag_pre + 'Streifenart_').text
            df_width.loc[i, 'fromWidth_cm'] = float(subchild.find(tag_pre + 'Von-Breite__cm__').text) / 100
            df_width.loc[i, 'toWidth_cm'] = float(subchild.find(tag_pre + 'Bis-Breite__cm__').text) / 100
            df_width.loc[i, 'fromNode'] = subchild.find(tag_pre + 'Von-Netzknoten__Von-Netzknoten_Von-Netzknoten_').text
            df_width.loc[i, 'fromNodeKey'] = subchild.find(tag_pre + 'Von-Netzknotenbuchstabe').text
            df_width.loc[i, 'toNode'] = subchild.find(tag_pre + 'Nach-Netzknoten_').text
            df_width.loc[i, 'toNodeKey'] = subchild.find(tag_pre + 'Nach-Netzknotenbuchstabe').text

            # Die Koordinaten des betrachteten Streckenabschnitts werden ausgelesen
            coordinates_unord = subchild.find(tag_pre + 'SHAPE')[0][0][0][0][0][0].text
            # Die ausgelesenen Koordinaten werden verarbeitet und zur Weiterverarbeitung angepasst

            if coordinates_unord[0] == ' ':
                coordinates_unord = coordinates_unord[1:]  # delete space character!
                coordinates_split = coordinates_unord.split(' ')
                longitude = coordinates_split[0::2]
                latitude = coordinates_split[1::2]
            else:
                coordinates_split = coordinates_unord.split(' ')
                longitude = coordinates_split[1::2]
                latitude = coordinates_split[0::2]

            # Format coordinates to float
            longitude = [float(x) for x in longitude]
            latitude = [float(x) for x in latitude]

            check_coordinate_value(longitude[0], latitude[0], 'baysis_laneWidth_to_pandas')

            df_width.loc[i, 'coordinatesEast'] = longitude
            df_width.loc[i, 'coordinatesNorth'] = latitude

        # Erhöhung der Iterationsvariablen
        i = i + 1


    # to_float = ['fromStation', 'toStation', 'roadLength']
    to_float = ['fromStation', 'toStation']
    to_int = ['fromNode', 'toNode', 'sectionNumber', 'fromWidth_cm', 'toWidth_cm', 'LaneStripeNumber', 'OBJECT ID']
    df_width = pandas_to_numeric(df_width, to_int, to_float, conv_index=True)

    df_width.sort_values(by=['roadName', 'OBJECT ID'], inplace=True)
    df_width.reset_index(drop=True, inplace=True)

    pa = paths.baysis_pd_lw_file
    df_width.to_pickle(pa)

    # Save failures to textfile
    with open(pa[:-4] + '_info.txt', 'w') as textFile:
        textFile.write('Lanes that Failed (probably) due to NoneType Error in Coordinates:\n')
        textFile.write(failIDs)  # TODO: fails not recognized correctly?

    # describe parse
    print('Done Parsing: ' + path_to_file)
    print('Saved in {}'.format(pa))
    print('View Errors in {}'.format(pa[:-4] + '_info.txt'))
    print('\nDataframe looks like:')
    print(df_width.head())


def baysis_lanes_to_pandas(path_to_file: str, gps=True, printFailures=0, new_format=True, no_save=False):
    """
        A quick parse of Baysis Ways to Pandas Dataframe
        ... print errors to console with printFailures
        ... Errors saved to Ways_info.txt
    """
    failIDs = ''

    # Erzeugung des XML-Baums
    tree = ET.parse(path_to_file)
    # Das Wurzelelement wird gespeichert
    root = tree.getroot()
    # Ein neuer DataFrame mit den notwendigen Spalten wird initialisiert
    df_number = pd.DataFrame(
        columns=['OBJECT ID', 'roadName', 'roadType', 'sectionNumber', 'fromStation', 'toStation',
                 'sameDirection', 'againstDirection', 'laneNumber', 'fromNode', 'toNode', 'fromNodeKey',
                 'toNodeKey', 'roadLength', 'coordinatesEast', 'coordinatesNorth'])
    tag_pre = '{https://www.baysis.bayern.de/gis/services/wfs/BAYSIS_Strassenbestand/MapServer/WFSServer}'
    # Erstellen einer Iterationsvariablen
    i = 0
    # For-Schleifen: Innerhalb der For-Schleifen werden die Unterstrukturen des XML-Files ausgelesen. Darin befinden sich die Informationen für jeden Streckenabschnitt der sich innerhalb
    # der betrachteten Bereichs befindet
    search_pattern = find_xml_search_pattern(root, 'lanes')
    for child in root.findall(search_pattern):
        for subchild in child:
            # Innerhalb einer Zeile des DataFrames werden die relevanten Informationen für einen definierten Streckenabschnitt gespeichert
            df_number.loc[i, 'OBJECT ID'] = subchild.find(tag_pre + 'OBJECTID').text
            df_number.loc[i, 'roadName'] = subchild.find(tag_pre + 'Straßenname_').text
            df_number.loc[i, 'roadType'] = subchild.find(tag_pre + 'Straßenklasse_').text
            df_number.loc[i, 'sectionNumber'] = subchild.find(tag_pre + 'Abschnittsnummer_').text
            df_number.loc[i, 'fromStation'] = float(subchild.find(tag_pre + 'Von-Station_').text)
            df_number.loc[i, 'toStation'] = float(subchild.find(tag_pre + 'Bis-Station_').text)
            df_number.loc[i, 'sameDirection'] = int(subchild.find(tag_pre + 'In-Richtung_').text)
            df_number.loc[i, 'againstDirection'] = int(subchild.find(tag_pre + 'Gegen-Richtung_').text)
            df_number.loc[i, 'laneNumber'] = int(subchild.find(tag_pre + 'Anzahl_').text)
            df_number.loc[i, 'fromNode'] = subchild.find(tag_pre + 'Von-Netzknoten').text
            df_number.loc[i, 'toNode'] = subchild.find(tag_pre + 'Nach-Netzknoten').text
            df_number.loc[i, 'fromNodeKey'] = subchild.find(tag_pre + 'Von-Netzknotenbuchstabe').text
            df_number.loc[i, 'toNodeKey'] = subchild.find(tag_pre + 'Nach-Netzknotenbuchstabe').text
            df_number.loc[i, 'roadLength'] = subchild.find(tag_pre + 'SHAPE.STLength__').text
            # Die Koordinaten des betrachteten Streckenabschnitts werden ausgelesen
            coordinates_unord = subchild.find(tag_pre + 'SHAPE')[0][0][0][0][0][0].text
            # Die ausgelesenen Koordinaten werden verarbeitet und zur weiterverarbeitung angepasst

            if coordinates_unord[0] == ' ':
                coordinates_unord = coordinates_unord[1:]  # delete space character!
                coordinates_split = coordinates_unord.split(' ')
                longitude = coordinates_split[0::2]
                latitude = coordinates_split[1::2]
            else:
                coordinates_split = coordinates_unord.split(' ')
                longitude = coordinates_split[1::2]
                latitude = coordinates_split[0::2]

            # Format coordinates to float
            longitude = [float(x) for x in longitude]
            latitude = [float(x) for x in latitude]

            check_coordinate_value(longitude[0], latitude[0], 'baysis_lanes_to_pandas')

            df_number.loc[i, 'coordinatesEast'] = np.asarray(longitude)
            df_number.loc[i, 'coordinatesNorth'] = np.asarray(latitude)

        # Erhöhung der Iterationsvariablen
        i = i + 1

    # Convert dictionary to Dataframe
    to_float = ['fromStation', 'toStation', 'roadLength']
    to_int = ['fromNode', 'toNode', 'sectionNumber', 'sameDirection', 'againstDirection', 'laneNumber', 'OBJECT ID']
    df_number = pandas_to_numeric(df_number, to_int, to_float, conv_index=True)

    df_number.sort_values(by=['roadName', 'OBJECT ID'], inplace=True)
    df_number.reset_index(drop=True, inplace=True)

    if no_save:
        return df_number
    else:
        if gps:
            pa = paths.baysis_pd_la_gps_file
        else:
            pa = paths.baysis_pd_la_krueger_file
        df_number.to_pickle(pa)

        # Save failures to textfile
        with open(pa[:-4] + '_info.txt', 'w') as textFile:
            textFile.write('Lanes that Failed (probably) due to NoneType Error in Coordinates:\n')
            textFile.write(failIDs)

        # describe parse
        print('Done Parsing: ' + path_to_file)
        print('Saved in {}'.format(pa))
        print('View Errors in {}'.format(pa[:-4] + '_info.txt'))
        print('\nDataframe looks like:')
        print(df_number.head())


def baysis_ways_to_pandas(path_to_file: str, printFailures=0, new_format=True):
    """
        A quick parse of Baysis Ways to Pandas Dataframe
        ... print errors to console with printFailures
        ... Errors saved to Ways_info.txt
    """

    # Local Variables
    failIDs = ''
    # Erzeugung des XML-Baums
    tree = ET.parse(path_to_file)
    # Das Wurzelelement wird gespeichert
    root = tree.getroot()
    # Ein neuer DataFrame mit den notwendigen Spalten wird initialisiert
    df_grid = pd.DataFrame(
        columns=['OBJECT ID', 'roadName', 'sectionID', 'sectionNumber', 'sectionNote', 'sectionLength',
                 'fromNode', 'fromNodeKey', 'fromNodeName', 'toNode', 'toNodeKey', 'toNodeName',
                 'streetID', 'sectionLength_2', 'roadType', 'coordinates'])

    tag_pre = '{https://www.baysis.bayern.de/gis/services/wfs/BAYSIS_Strassennetz/MapServer/WFSServer}'
    # Erstellen einer Iterationsvariablen
    i = 0
    # For-Schleifen: Innerhalb der For-Schleifen werden die Unterstrukturen des XML-Files ausgelesen. Darin befinden sich die Informationen für jeden Streckenabschnitt der sich innerhalb
    # der betrachteten Bereichs befindet
    search_pattern = find_xml_search_pattern(root, 'ways')
    for child in root.findall(search_pattern):
        for subchild in child:
            # Innerhalb einer Zeile des DataFrames werden die relevanten Informationen für einen definierten Streckenabschnitt gespeichert

            df_grid.loc[i, 'OBJECT ID'] = subchild.find(tag_pre + 'OBJECTID').text
            df_grid.loc[i, 'roadName'] = subchild.find(tag_pre + 'Straßenname_').text
            df_grid.loc[i, 'sectionID'] = subchild.find(tag_pre + 'Abschnitts-ID_').text
            if subchild.find(tag_pre + 'Abschnittsnummer_') is not None:
                df_grid.loc[i, 'sectionNumber'] = subchild.find(tag_pre + 'Abschnittsnummer_').text
            else:
                df_grid.loc[i, 'sectionNumber'] = None
            df_grid.loc[i, 'sectionNote'] = subchild.find(tag_pre + 'Abschnittsanzeige_').text
            df_grid.loc[i, 'sectionLength'] = subchild.find(tag_pre + 'SHAPE.STLength__').text
            df_grid.loc[i, 'fromNode'] = subchild.find(tag_pre + 'Von-Netzknoten_').text
            df_grid.loc[i, 'fromNodeKey'] = subchild.find(tag_pre + 'Von-Netzknotenbuchstabe_').text
            df_grid.loc[i, 'fromNodeName'] = subchild.find(tag_pre + 'Von-Netzknotenname_').text
            df_grid.loc[i, 'toNode'] = subchild.find(tag_pre + 'Nach-Netzknoten_').text
            df_grid.loc[i, 'toNodeKey'] = subchild.find(tag_pre + 'Nach-Netzknotenbuchstabe_').text
            df_grid.loc[i, 'toNodeName'] = subchild.find(tag_pre + 'Nach-Netzknotenname_').text
            df_grid.loc[i, 'streetID'] = subchild.find(tag_pre + 'Straßen-ID_').text
            df_grid.loc[i, 'sectionLength_2'] = float(subchild.find(tag_pre + 'Abschnittslänge_').text)
            df_grid.loc[i, 'roadType'] = subchild.find(tag_pre + 'Straßenklasse_').text

            # Die Koordinaten des betrachteten Streckenabschnitts werden ausgelesen
            coordinates_unord = subchild.find(tag_pre + 'SHAPE')[0][0][0][0].text
            # Die ausgelesenen Koordinaten werden verarbeitet und zur Weiterverarbeitung angepasst
            coordinates_split = coordinates_unord.split(' ')
            coordinates_split = [float(x) for x in coordinates_split]
            df_grid.loc[i, 'coordinates'] = coordinates_split

        # Erhöhung der Iterationsvariablen
        i = i + 1

    to_float = ['sectionLength']
    to_int = ['fromNode', 'toNode', 'sectionNumber', 'OBJECT ID']
    df_grid = pandas_to_numeric(df_grid, to_int, to_float, conv_index=True)
    df_grid.sort_values(by=['roadName', 'OBJECT ID'], inplace=True)
    df_grid.reset_index(drop=True, inplace=True)

    # Save Dataframe as pickle
    pa = paths.baysis_pd_wa_file
    df_grid.to_pickle(pa)
    with open(pa[:-4] + '_info.txt', 'w') as textFile:
        textFile.write('Ways that Failed (probably) due to NoneType Error in Coordinates:\n')
        textFile.write(failIDs)

    # describe parse
    print('Done Parsing: ' + path_to_file)
    print('Saved in {}'.format(pa))
    print('View Errors in {}'.format(pa[:-4] + '_info.txt'))
    print('\nDataframe looks like:')
    print(df_grid.head())


def baysis_nodes_to_pandas(path_to_file: str, printFailures=0, new_format=True):
    """
        A quick parse of Baysis Nodes to Pandas Dataframe
        ... print errors to console with printFailures
        ... Errors saved to Nodes_info.txt
        new_format = True for updated Baysis Database
    """
    # Temporary Variables
    failIDs = ''

    # Erzeugung des XML-Baums
    tree = ET.parse(path_to_file)
    # Das Wurzelelement wird gespeichert
    root = tree.getroot()
    # Ein neuer DataFrame mit den notwendigen Spalten wird initialisiert
    df_node = pd.DataFrame(
        columns=['lon', 'lat', 'Type', 'OBJECT ID', 'Node', 'TypeKey', 'TypeName'])
    tag_pre = '{https://www.baysis.bayern.de/gis/services/wfs/BAYSIS_Strassennetz/MapServer/WFSServer}'
    # Erstellen einer Iterationsvariablen
    i = 0
    # For-Schleifen: Innerhalb der For-Schleifen werden die Unterstrukturen des XML-Files ausgelesen. Darin befinden sich die Informationen für jeden Streckenabschnitt der sich innerhalb
    # der betrachteten Bereichs befindet
    search_pattern = find_xml_search_pattern(root, 'nodes')
    for child in root.findall(search_pattern):
        for subchild in child:
            # Innerhalb einer Zeile des DataFrames werden die relevanten Informationen für einen definierten Streckenabschnitt gespeichert
            df_node.loc[i, 'OBJECT ID'] = subchild.find(tag_pre + 'OBJECTID').text
            df_node.loc[i, 'Node'] = subchild.find(tag_pre + 'Netzknoten_').text
            df_node.loc[i, 'TypeKey'] = subchild.find(tag_pre + 'Netzknotentyp__Schlüssel__').text
            df_node.loc[i, 'Type'] = subchild.find(tag_pre + 'Netzknotentyp_').text
            df_node.loc[i, 'TypeName'] = subchild.find(tag_pre + 'Netzknotenname_').text

            # Die Koordinaten des betrachteten Streckenabschnitts werden ausgelesen
            coordinates_unord = subchild.find(tag_pre + 'SHAPE')[0][0].text
            # Die ausgelesenen Koordinaten werden verarbeitet und zur Weiterverarbeitung angepasst
            coordinates_split = coordinates_unord.split(',')
            if len(coordinates_split) < 2:
                coordinates_split = coordinates_unord.split(' ')
                if len(coordinates_split) < 2:
                    print(f"Could not split coordinates from {coordinates_unord}")
                    exit()
                # gps in format [north, east]
                latitude = coordinates_split[0]
                longitude = coordinates_split[1]
            else:
                # gps in format [east, north]
                latitude = coordinates_split[1]
                longitude = coordinates_split[0]

            check_coordinate_value(longitude, latitude, 'baysis_nodes_to_pandas')

            df_node.loc[i, 'lon'] = longitude
            df_node.loc[i, 'lat'] = latitude
        # Erhöhung der Iterationsvariablen
        i = i + 1

    to_float = ['lon', 'lat']
    to_int = ['OBJECT ID']
    df_node = pandas_to_numeric(df_node, to_int, to_float, conv_index=True)
    df_node.sort_values(by='OBJECT ID', inplace=True)
    df_node.reset_index(drop=True, inplace=True)

    # save dataframe as pickle
    pa = paths.baysis_pd_no_file
    df_node.to_pickle(pa)
    with open(pa[:-4] + '_info.txt', 'w') as textFile:
        textFile.write('Nodes that Failed due to NoneType Error in Coordinates:\n')
        textFile.write(failIDs)

    # describe parse
    print('Done Parsing: ' + path_to_file)
    print('Saved in {}'.format(pa))
    print('Dataframe looks like:')
    print(df_node.head())
    print('View Errors in {}'.format(pa[:-4] + '_info.txt'))


def check_coordinate_value(longitude: str, latitude: str, name: str) -> None:
    try:
        if not 8 < float(longitude) < 15 or not 45 < float(latitude) < 52:
            print(f"Waring in {name}: longitude and latitude might be inverted.")
    except ValueError:
        print(f"Error in {name}: Could not convert coordinates string to float: {longitude} and {latitude}")


def baysis_lanes_combine_Kruger_GPS():
    """
        This simply puts the Krueger and GPS data into one Pandas Dataframe
        and saves them as pandas file
    """
    with open(paths.baysis_pd_la_krueger_file, 'rb') as package:
        lanes_krueger = pickle.load(package)
    with open(paths.baysis_pd_la_gps_file, 'rb') as package:
        lanes_gps = pickle.load(package)

    lanes = lanes_krueger
    lanes['longitude'] = lanes_gps['coordinatesEast']
    lanes['latitude'] = lanes_gps['coordinatesNorth']
    lanes.to_pickle(paths.baysis_pd_la_file)

