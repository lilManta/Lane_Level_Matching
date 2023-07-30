import math
import numpy as np
from geographiclib.geodesic import Geodesic

#####################################################
## This script determines various distances between
## coordinates and angles
#####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
#####################################################
## Date: 06.2020
#####################################################


def geoDistancePythagoras(lat_1, lon_1, lat_2, lon_2):
    """
    Die Funktion dient der Berechnung des Abstandes zweier nahe beieinander gelegener Punkte
    Annahme der ebenen Fläche nur für kurze Abstände gültig, sonst: Orthonome
    :param lat_1:
    :param lon_1:
    :param lat_2:
    :param lon_2:
    :return: distance in meters
    """
    # Berechnung des mittleren Breitengrads in rad
    lat_m_rad=np.deg2rad((lat_2+lat_1)/2)
    # Bestimmung der Abständige in Längen- und Breitenrichtung
    dx=111300 * np.cos(lat_m_rad) * (lon_2-lon_1)
    dy=111300 * (lat_2-lat_1)
    # Berechnung des absoluten Abstands in METER
    d=np.sqrt(dx**2+dy**2)

    return d


def directionCalculation(lat_1, lon_1, lat_2, lon_2):
    """
    Die Funktion dient der Ermittlung der Orientierung,
    die beim Zurücklegen der Strecke zwischen zwei Koordinatenpunkten vorherrscht
    0° entspricht Norden, 90° Osten, usw.
    Limit bei -180° bis 180°
    :param lat_1:   latitude first coordinate in WGS84
    :param lon_1:   longitude first coordinate in WGS84
    :param lat_2:   latitude second coordinate in WGS84
    :param lon_2:   longitude second coordinate in WGS84
    :return:        orientation
    """

    if lat_1 == lat_2 and lon_1 == lon_2:
        # same coordinates
        return None
    geod = Geodesic.WGS84
    orientation=geod.Inverse(lat_1, lon_1,lat_2, lon_2)
    orientation=float(orientation['azi1'])
    if orientation < 0:
        orientation = 360 + orientation
    return orientation


def get_orient_diff(orient1: float, orient2: float) -> float:
    diff = abs(orient1 - orient2)
    if diff > 180:
        diff = 360 - diff
    return diff
