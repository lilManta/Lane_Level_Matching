import pandas as pd
import numpy as np
import math as m
from utils.DetermineDistance import geoDistancePythagoras, directionCalculation, get_orient_diff

from progressbar import ProgressBar
pbar = ProgressBar()

####################################################
## This script extracts the baysis street information
## from each downloaded chapter and combines them in
## one street map saved as pandas dataframe
####################################################
## Concept: Bertram Fuchs, FTM
## Edit: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################


def connectGridNumber(df_grid, df_number):
    """
    This function connects the dataframe Ways with the dataframe Lanes
    by grouping the nodes.

    :param df_grid: Ways dataframe from BAYSIS
    :param df_number: Lanes dataframe from BAYSIS
    :return: dataframe with all map important information
    """
    columns_connectGridNumber = ['latitude', 'longitude', 'roadName', 'roadType', 'coordinateDistance',
                                 'cumulativeDistance', 'orientation', 'fromNode', 'toNode', 'fromNodeKey',
                                 'toNodeKey', 'sectionLength', 'sectionNumber', 'fromStationWidth', 'toStationWidth',
                                 'sameDirection', 'againstDirection', 'lanesTotal', 'fromWidth_cm',
                                 'toWidth_cm', 'offsetSameDir', 'offsetAgainstDir']

    df_dataset = pd.DataFrame(columns=columns_connectGridNumber)
    # Ein Group-Objekt wird erstellt -> Gruppierung erfolgt nach Netzknoten und Netzknotenbuchstaben
    grid_grouped = df_grid.groupby(['fromNode', 'fromNodeKey', 'toNode', 'toNodeKey'])

    # For-Schleife: Verknüpfung von Straßenverlaufsinformationen und Fahrspurinformationen
    # for name, group in grid_grouped:
    for name, group in pbar(grid_grouped):
        # Erster Index der Gruppe wird extrahiert -> Da alle Gruppen nur jeweils ein member-Eintrag enthalten, entspricht der erste Index dem des einzigen Elements
        index_group = group.index[0]

        # Aufteilung der Koordinaten-Liste in latitude- und longitude-Komponente
        # check baysis format
        if group.loc[index_group, 'coordinates'][0] > 25:
            # new baysis format latitude first (09.2020 - ?)
            longitude = group.loc[index_group, 'coordinates'][1::2]
            latitude = group.loc[index_group, 'coordinates'][0::2]
        else:
            # old baysis format longitude first (05.2020 - 08.2020)
            longitude = group.loc[index_group, 'coordinates'][0::2]
            latitude = group.loc[index_group, 'coordinates'][1::2]

        # Neue, leere Liste -> Dient der Aufnahme der Abstände zwischen den Koordinatenpunkten
        distance_list = [0]

        # Neue, leere Liste -> Dient der Aufnahme von Zeitstempeln -> Die Distanzen zwischen den Koordinatenpunkten werden in Metern berechnet -> Der Abstandswert wird in einen Zeitstempel umgewandelt
        # Um eine Gleichmäßige Verteilung der Stützstellen zu erreichen, wird der Datensatz später geresampled -> Große Lücken werden somit verkleinert (Zeitstempel werden wegen Resampling erstellt)
        time_list = [pd.to_datetime(0, unit='s')]

        # For-Schleife: Alle ursprünglichen Koordinatenpunkte des aktuell betrachteten Straßenabschnitts werden durchlaufen und die jeweils dazwischenliegenden Abstände berechnet
        for i in range(1, len(latitude)):
            # Berechnung der geometrischen Abstände
            distance_neighbors = geoDistancePythagoras(latitude[i], longitude[i], latitude[i - 1], longitude[i - 1])
            distance_list.append(distance_neighbors)
            # Aufsummieren der Abstände
            distance_cum = sum(distance_list)
            # Umwandlung in Zeitstempel
            timestamp = pd.to_datetime(distance_cum, unit='s')
            # Zeitstempel werden in Liste gespeichert
            time_list.append(timestamp)

        # Ein temporärer DataFrame wird erstellt und mit den entsprechenden Informationen aus dem DataFrame "df_grid" befüllt -> Länge des DataFrames = Anzahl der Koordinatenpunkte
        df_temp = pd.DataFrame(index=range(0, len(time_list)), columns=columns_connectGridNumber)

        # Indizierung des temporären DataFrames
        df_temp.index = time_list

        # Resampling der Daten -> Zwischenwerte werden Interpoliert -> Damit wird die Zahl der Stützstellen/Koordinatenpunkte erhöht
        df_temp['latitude'] = latitude
        df_temp['longitude'] = longitude
        df_temp_resampled = df_temp.resample('S').ffill()

        nan_list = df_temp_resampled.loc[:, 'latitude'].duplicated(keep='first')
        nan_list2 = df_temp_resampled.loc[:, 'longitude'].duplicated(keep='first')
        nan_list = nan_list.combine(nan_list2, min)

        df_temp_resampled.loc[nan_list, 'latitude'] = np.nan
        df_temp_resampled.loc[nan_list, 'longitude'] = np.nan
        df_temp_resampled.iloc[-1, 0] = latitude[-1]
        df_temp_resampled.iloc[-1, 1] = longitude[-1]
        df_temp_resampled.latitude = df_temp_resampled.latitude.interpolate()
        df_temp_resampled.longitude = df_temp_resampled.longitude.interpolate()

        # Befülluung der DataFrames mit weiteren relevanten Daten
        df_temp_resampled = df_temp_resampled.reset_index(drop=True)
        df_temp_resampled.iloc[-1, 0] = latitude[-1]
        df_temp_resampled.iloc[-1, 1] = longitude[-1]
        df_temp_resampled.loc[:, 'fromNode'] = group.loc[index_group, 'fromNode']
        df_temp_resampled.loc[:, 'toNode'] = group.loc[index_group, 'toNode']
        df_temp_resampled.loc[:, 'fromNodeKey'] = group.loc[index_group, 'fromNodeKey']
        df_temp_resampled.loc[:, 'toNodeKey'] = group.loc[index_group, 'toNodeKey']
        df_temp_resampled.loc[:, 'sectionLength'] = group.loc[index_group, 'sectionLength']
        df_temp_resampled.loc[:, 'coordinateDistance'] = df_temp_resampled.index[-1] / (
                df_temp_resampled.loc[0, 'sectionLength'] * 1000)
        df_temp_resampled.loc[0, 'cumulativeDistance'] = 0
        df_temp_resampled.loc[1:, 'cumulativeDistance'] = df_temp_resampled.index[-1] / (
                df_temp_resampled.loc[0, 'sectionLength'] * 1000)
        df_temp_resampled.loc[:, 'cumulativeDistance'] = df_temp_resampled.loc[:, 'cumulativeDistance'].cumsum()
        df_temp_resampled.loc[:, 'roadType'] = group.loc[index_group, 'roadType']
        df_temp_resampled.loc[:, 'roadName'] = group.loc[index_group, 'roadName']
        df_temp_resampled.loc[:, 'sectionNumber'] = group.loc[index_group, 'sectionNumber']

        # Berechnung der Orientierung für Teilsegmente gleicher Länge eines Straßenabschnitts -> Über 50 Indizes wird die "Steigung"/Orientierung berechnet
        nan_list = nan_list.reset_index(drop=True)
        idx_orientation_change = nan_list.index[nan_list == False].tolist()
        if idx_orientation_change[-1] != nan_list.index[-1]:
            idx_orientation_change.append(nan_list.index[-1])
        orientation_last = None
        for idx_oc in range(len(idx_orientation_change)-1):
            i1 = idx_orientation_change[idx_oc]
            i2 = idx_orientation_change[idx_oc+1]
            orientation = directionCalculation(df_temp_resampled.loc[i1, 'latitude'],
                                               df_temp_resampled.loc[i1, 'longitude'],
                                               df_temp_resampled.loc[i2, 'latitude'],
                                               df_temp_resampled.loc[i2, 'longitude'])
            if orientation is None and orientation_last is None:
                # start segment can be at the same coordinate for a while
                orientation = 0
            if orientation_last is None:
                # write orientation
                df_temp_resampled.loc[i1:i2, 'orientation'] = float(orientation)
                orientation_last = orientation
            elif orientation is None:
                # write last known orientation
                df_temp_resampled.loc[i1:i2, 'orientation'] = float(orientation_last)
            elif get_orient_diff(orientation_last, orientation) < 30:
                # write orientation
                df_temp_resampled.loc[i1:i2, 'orientation'] = float(orientation)
                orientation_last = orientation
            elif get_orient_diff(orientation_last, orientation):
                # write orientation but reduce difference with last orientation
                df_temp_resampled.loc[i1:i2, 'orientation'] = float((2*orientation + orientation_last)/3)
                orientation_last = orientation
            else:
                # exclude jumping coordinate in orientation and wait till orientation last is near again
                df_temp_resampled.loc[i1:i2, 'orientation'] = float(orientation_last)
                orientation_last = (orientation + 3*orientation_last)/4

        attr = [group.loc[index_group, 'fromNode'], group.loc[index_group, 'toNode'],
                group.loc[index_group, 'fromNodeKey'], group.loc[index_group, 'toNodeKey']]

        # Im DataFrame "df_number" wird anhand der Attribute der zugehörige Straßenabschnitt mit den Fahrstreifeninformationen herausgefiltert
        street = df_number.loc[(df_number['fromNode'] == attr[0]) & (df_number['toNode'] == attr[1]) & (
                df_number['fromNodeKey'] == attr[2]) & (df_number['toNodeKey'] == attr[3]), :]
        segment = street.groupby(['fromStation', 'toStation'])

        # Ist kein Straßenabschnitt im DataFrame hinterlegt ("street" ist leer) -> es handelt sich um eine Zubringerstraße/Auffahrt etc., zu diesen Elementen sind keine Fahrspurinformationen vorhanden
        if street.empty == True:
            df_temp_resampled.loc[:, 'lanesTotal'] = 1
            df_temp_resampled.loc[:, 'sameDirection'] = 1
            df_temp_resampled.loc[:, 'againstDirection'] = 0
            df_temp_resampled.loc[:, 'fromStationWidth'] = 0
            df_temp_resampled.loc[:, 'toStationWidth'] = group.loc[index_group, 'sectionLength']

        # Ist die Variable "street" nicht leer -> Informationen zum aktuellen Straßenabschnitt sind vorhanden -> Anzahl der Fahrspuren kann ermittelt werden
        else:
            east_diff = street.loc[street.index[0], 'coordinatesEast'][0] - df_temp_resampled.loc[0, 'longitude']
            north_diff = street.loc[street.index[0], 'coordinatesNorth'][0] - df_temp_resampled.loc[0, 'latitude']
            offset_sameDir = [east_diff, north_diff]
            if len(street) > 1:
                east_diff = street.loc[street.index[1], 'coordinatesEast'][0] - df_temp_resampled.loc[0, 'longitude']
                north_diff = street.loc[street.index[1], 'coordinatesNorth'][0] - df_temp_resampled.loc[0, 'latitude']
                offset_againstDir = [east_diff, north_diff]
            else:
                offset_againstDir = offset_sameDir
            df_temp_resampled.loc[:, 'offset_sameDir_lon'] = offset_sameDir[0]
            df_temp_resampled.loc[:, 'offset_sameDir_lat'] = offset_sameDir[1]

            df_temp_resampled.loc[:, 'offset_againstDir_lon'] = offset_againstDir[0]
            df_temp_resampled.loc[:, 'offset_againstDir_lat'] = offset_againstDir[1]
            # new

            index_lower = 0
            last_element = False
            # For-Schleife: Dient dem Durchlauf der Teilsegmente eines Straßenabschnitts -> Wechselt die Fahrspuranzahl innerhalb eines Straßenabschnitts einfach/mehrfach, sind die jeweils zu den Teilsegment gehörenden Fahrspuranzahlen separat im DataFrame df_number vorhanden
            for name, subgroup in segment:
                if len(name) > 2:
                    print("Lane number change within one segment found")
                else:
                    last_element = True

                # Zwischenspeicherung des ersten Indizes der subgroup
                index_subgroup = subgroup.index[0]

                # Berechnung der oberen Indexgrenze -> Die Länge des DataFrames df_temp_resampled entspricht (ca.) der Länge des gesamten Straßenabschnitts in Metern -> Bei mehreren Segmenten wird immer obere Index-Wert über das Ende des Teilsegments ermittelt

                if last_element:
                    index_upper = len(df_temp_resampled) - 2
                else:
                    index_upper = int(  # m.floor(
                        df_temp_resampled.index[-1] *  # df_temp_resampled.loc[0, 'coordinateDistance'] *
                        (subgroup.loc[index_subgroup, 'toStation'] / df_temp_resampled.loc[0, 'sectionLength']))

                # Die Fahrspur- und Abschnittsinformationen des aktuell betrachteten Teilsegments werden dem DataFrame df_temp_resampled übergeben
                df_temp_resampled.loc[index_lower:(index_upper + 1), 'fromStationWidth'] = subgroup.loc[
                    index_subgroup, 'fromStation']
                df_temp_resampled.loc[index_lower:(index_upper + 1), 'toStationWidth'] = subgroup.loc[
                    index_subgroup, 'toStation']
                df_temp_resampled.loc[index_lower:(index_upper + 1), 'sameDirection'] = subgroup.loc[
                    index_subgroup, 'sameDirection']
                df_temp_resampled.loc[index_lower:(index_upper + 1), 'againstDirection'] = subgroup.loc[
                    index_subgroup, 'againstDirection']
                df_temp_resampled.loc[index_lower:(index_upper + 1), 'lanesTotal'] = int(
                    subgroup.loc[index_subgroup, 'sameDirection']) + int(
                    subgroup.loc[index_subgroup, 'againstDirection'])

                # Aktualisierung des unteren Indizes
                index_lower = index_upper + 1
        # Der temporäre DataFrame df_temp_resampled wird dem Gesamtdatensatz hinzugefügt -> df_dataset enthält am Ende der Funktion die Koordinaten des gesamten Straßenverlaufs des betrachteten Ausschnitts, sowie weitere relevante Informationen
        # Concatenate coordinates in one row
        df_dataset = df_dataset.append(df_temp_resampled, sort=False, ignore_index=True)

    # Convert to types
    df_dataset = df_dataset.astype({'cumulativeDistance': float, 'orientation': float, 'fromNode': int, 'toNode': int,
                                    'sameDirection': 'int8', 'againstDirection': 'int8', 'lanesTotal': 'int8'})

    return df_dataset


def connectGridNumberWidth(df_mapData, df_width):
    """
    This function connects the dataframe from connectGridNumber with the lanewidth dataframe
    unnecessary data like lanewidths of bicycle roads are removed

    :param df_mapData:  dataframe from function connectGridNumber()
    :param df_width:    dataframe LaneWidth from BAYSIS
    :return:            dataframe with all map important information
    """
    # Die Funktion dient dazu den mithilfe der Funktion "connectGridNumber" zuvor erstellten DataFrame mit df_width zu verknüpfen. Der daraus resultierende DataFrame
    # enthält dann zu jeder in Baysis hinterlegten Straße innerhalb des definierten Bereichs die Fahrspurinformationen sowie die Fahrbahnbreiten
    # Der DateFrame df_width wird um die überflüssigen Einträge reduziert -> Breiten von Seitenstreifen, Ausbuchtungen, etc. nicht notwendig
    df_width_new = df_width.loc[(df_width['laneStripeType'] == 'Fahrbahn') | \
                                df_width['laneStripeType'].str.contains('Mittelstreifen', regex=False) | \
                                df_width['laneStripeType'].str.contains('Brücken', regex=False) | \
                                df_width['laneStripeType'].str.contains('Trennstreifen', regex=False) | \
                                df_width['laneStripeType'].str.contains('Kreuzungsüberfahrt', regex=False) | \
                                df_width['laneStripeType'].str.contains('Verkehrsinsel', regex=False), :]

    # Dem DataFrame df_mapData werden zwei neue Spalten hinzugefügt -> Aufnahme der Von-/Bis-Station Werte für die Fahrbahnbreiten
    df_mapData.insert(12, 'fromStation', 0)
    df_mapData.insert(13, 'toStation', df_mapData.loc[:, 'sectionLength'])
    # Umbenennung der vorhandenen Spalten Von-Station und Bis-Station
    df_mapData = df_mapData.rename(columns={'fromStation': 'fromStationNumber', 'toStation': 'toStationNumber'})

    # Ein Group-Objekt wird erstellt -> Gruppierung erfolgt nach Netzknoten und Netzknotenbuchstaben
    dataset_grouped = df_mapData.groupby(['fromNode', 'fromNodeKey', 'toNode', 'toNodeKey'], sort=False)
    # For-Schleife: Verknüpfung von Straßenverlaufsinformationen und Fahrspurinformationen
    for name, group in dataset_grouped:
        # Erster und letzter Index der Gruppe werden extrahiert -> Dient der Berechnung der Gruppenlännge
        index_group_first = group.index[0]
        index_group_last = group.index[-1]

        # Zwischenspeicherung der Attribute fromNode, toNode, fromNodeKey und toNodeKey des aktuellen Group-Elements
        attr = [group.loc[index_group_first, 'fromNode'], group.loc[index_group_first, 'toNode'],
                group.loc[index_group_first, 'fromNodeKey'], group.loc[index_group_first, 'toNodeKey']]

        # Im DataFrame "df_width_new" wird anhand der Attribute der zugehörige Straßenabschnitt mit den Fahrstreifeninformationen herausgefiltert
        street = df_width_new.loc[(df_width_new['fromNode'] == attr[0]) & (df_width_new['toNode'] == attr[1]) & (
                df_width_new['fromNodeKey'] == attr[2]) & (df_width_new['toNodeKey'] == attr[3]), :]

        # Der zuvor erstellte DataFrame "street" wird nach den Attributen "Von-Station" und "Bis-Station gruppiert" -> so erhält man alle Teilsegmente des Straßenabschnitts
        segment = street.groupby(['fromStation', 'toStation'])

        # Falls der reduzierte DataFrame "street" keine Einträge enthält -> Eintritt -> Dann sind zu diesem Abschnitt keine Fahrbahnbreiten hinterlegt
        if street.empty == True:
            # Befüllung der entsprechenden Zeilen in df_mapData mit "genormten" Werten
            df_mapData.loc[index_group_first:(index_group_last + 1), 'fromWidth_cm'] = 4
            df_mapData.loc[index_group_first:(index_group_last + 1), 'toWidth_cm'] = 0
            df_mapData.loc[index_group_first:(index_group_last + 1), 'offsetSameDir'] = 0
            df_mapData.loc[index_group_first:(index_group_last + 1), 'offsetAgainstDir'] = 0
            df_mapData.loc[index_group_first:(index_group_last + 1), 'fromStationWidth'] = 0
            df_mapData.loc[index_group_first:(index_group_last + 1), 'toStationWidth'] = group.loc[
                index_group_first, 'sectionLength']

        # Sind Fahrbahnbreiten für den aktuell betrachteten Straßenabschnitt vorhanden, erfolgt der Eintritt in die Else-Bedingung -> Zuordnung der Fahrbahnbreiten zu den entsprechenden Teilsegmenten innerhalb des Straßenabschnitts
        else:
            # Der Variable 'index_lower' wird der Index-Wert des ersten Eintrags der aktuellen Gruppe übergeben -> Betrachtung des aktuellen Segments im DataFrame df_mapData
            index_lower = index_group_first

            # For-Schleife: Dient dem Durchlaufen der Gruppen innerhalb des GroupBy-Objects -> subgroup entspricht Teilsegment eines Straßenabschnitts
            for name, subgroup in segment:
                # Erster Index der Subgroup wird in Variable "index_subgroup_first abgelegt"
                index_subgroup_first = subgroup.index[0]
                # Berechnung des oberen Indizes der subgroup -> Abschnittslänge des Teilsegments
                index_upper = m.floor((((index_group_last - index_group_first + 1) / group.loc[
                    index_group_first, 'sectionLength']) * (
                                           subgroup.loc[index_subgroup_first, 'toStation']))) + index_group_first

                # Enthält die subgroup nur einen Eintrag -> Eintritt -> Breite ist für beide Fahrstreifen gemeinsam angegeben -> kein zwischenelement trennt die Spuren
                if len(subgroup) == 1:
                    # Zuweiseung der Fahrbahnbreiten in beide Fahrtrichtungen
                    df_mapData.loc[index_lower:(index_upper + 1), 'fromWidth_cm'] = subgroup.loc[
                                                                                        index_subgroup_first, 'toWidth_cm'] / 2
                    df_mapData.loc[index_lower:(index_upper + 1), 'toWidth_cm'] = subgroup.loc[
                                                                                      index_subgroup_first, 'toWidth_cm'] / 2
                    # Zuweisung des Offsets von der Mittellinie des Straßenabschnitts
                    df_mapData.loc[index_lower:(index_upper + 1), 'offsetSameDir'] = 0
                    df_mapData.loc[index_lower:(index_upper + 1), 'offsetAgainstDir'] = 0
                    # Zuweisung der Teilsegment-Stationen -> In welchem Teil des Straßenabschnitts gelten diese Fahrbahnbreiten?
                    df_mapData.loc[index_lower:(index_upper + 1), 'fromStationWidth'] = subgroup.loc[
                        index_subgroup_first, 'fromStation']  # TODO
                    df_mapData.loc[index_lower:(index_upper + 1), 'toStationWidth'] = subgroup.loc[
                        index_subgroup_first, 'toStation']  # TODO

                if len(subgroup) == 2:
                    # Die Indizes der Einträge für den linken bzw. rechten Fahrstreifen werden gespeichert
                    index_width_R = subgroup[
                        (subgroup['laneStripeType'] == 'Fahrbahn') & (subgroup['laneStripe'] == 'R')].index
                    index_width_L = subgroup[
                        (subgroup['laneStripeType'] == 'Fahrbahn') & (subgroup['laneStripe'] == 'L')].index
                    # Workaround -> manche Einträge unvollständig
                    if len(index_width_L) == 0 or len(index_width_R) == 0:
                        continue
                    # Zuweisung der Fahrbahnbreiten in beide Fahrtrichtungen
                    df_mapData.loc[index_lower:(index_upper + 1), 'fromWidth_cm'] = subgroup.loc[
                        index_width_R[0], 'toWidth_cm']
                    df_mapData.loc[index_lower:(index_upper + 1), 'toWidth_cm'] = subgroup.loc[
                        index_width_L[0], 'toWidth_cm']
                    # Zuweisung des Offsets von der Mittellinie des Straßenabschnitts
                    df_mapData.loc[index_lower:(index_upper + 1), 'offsetSameDir'] = 0
                    df_mapData.loc[index_lower:(index_upper + 1), 'offsetAgainstDir'] = 0
                    # Zuweisung der Teilsegment-Stationen -> In welchem Teil des Straßenabschnitts gelten diese Fahrbahnbreiten?
                    df_mapData.loc[index_lower:(index_upper + 1), 'fromStationWidth'] = subgroup.loc[
                        index_subgroup_first, 'fromStation']
                    df_mapData.loc[index_lower:(index_upper + 1), 'toStationWidth'] = subgroup.loc[
                        index_subgroup_first, 'toStation']

                elif len(subgroup) >= 3:
                    # Die Indizes der Einträge für den linken, den rechten und den Mittelstreifen werden gespeichert
                    index_width_R = subgroup.loc[(subgroup.loc[:, 'laneStripeType'] == 'Fahrbahn') & (
                            subgroup.loc[:, 'laneStripe'] == 'R'), 'toWidth_cm'].index
                    index_width_L = subgroup.loc[(subgroup.loc[:, 'laneStripeType'] == 'Fahrbahn') & (
                            subgroup.loc[:, 'laneStripe'] == 'L'), 'toWidth_cm'].index
                    index_width_M = subgroup.loc[subgroup['laneStripe'] == 'M', 'toWidth_cm'].index
                    # Workaround -> manche Einträge unvollständig
                    if len(index_width_L) == 0 or len(index_width_R) == 0 or len(index_width_M) == 0:
                        continue
                    # Zuweisung der Fahrbahnbreiten in beide Fahrtrichtungen
                    df_mapData.loc[index_lower:(index_upper + 1), 'fromWidth_cm'] = subgroup.loc[
                        index_width_R[0], 'toWidth_cm']
                    df_mapData.loc[index_lower:(index_upper + 1), 'toWidth_cm'] = subgroup.loc[
                        index_width_L[0], 'toWidth_cm']
                    # Zuweisung des Offsets von der Mittellinie des Straßenabschnitts
                    df_mapData.loc[index_lower:(index_upper + 1), 'offsetSameDir'] = subgroup.loc[index_width_M[
                                                                                                      0], 'toWidth_cm'] / 2
                    df_mapData.loc[index_lower:(index_upper + 1), 'offsetAgainstDir'] = subgroup.loc[index_width_M[
                                                                                                         0], 'toWidth_cm'] / 2
                    # Zuweisung der Teilsegment-Stationen -> In welchem Teil des Straßenabschnitts gelten diese Fahrbahnbreiten?
                    df_mapData.loc[index_lower:(index_upper + 1), 'fromStationWidth'] = subgroup.loc[
                        index_subgroup_first, 'fromStation']
                    df_mapData.loc[index_lower:(index_upper + 1), 'toStationWidth'] = subgroup.loc[
                        index_subgroup_first, 'toStation']
                # Der untere Index wird aktualisiert -> nächste Subgroup beginnt im DataFrame eine Stelle nach der letzten
                index_lower = index_upper + 1

    # Set column types
    df_mapData = df_mapData.astype({'fromWidth_cm': float, 'toWidth_cm': float,
                                    'offsetSameDir': float, 'offsetAgainstDir': float})

    return df_mapData
