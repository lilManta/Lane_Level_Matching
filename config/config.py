####################################################
## This script contains all configuration parameters
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 05.2020
####################################################

#######################
# Baysis title requests
#######################
baysis_version = '2.0.0'

type_service_baysis = ['BAYSIS_Strassenbestand', 'BAYSIS_Strassennetz']
type_name_baysis_bestand = ['STRBESTAND_WFS:Fahrstreifen', 'STRBESTAND_WFS:Fahrbahnbreiten']
type_name_baysis_netz = ['STRNETZ_WFS:Strassennetz', 'STRNETZ_WFS:Netzknoten', 'STRNETZ_WFS:Stationierung_Hatches100']
type_coordinate_baysis = ['UTM', 'Krueger', 'WGS84']  # Krueger out of date?!
type_coordinate_baysis_code = ['urn:ogc:def:crs:EPSG:6.9:25832', 'urn:ogc:def:crs:EPSG:6.9:31468',
                               'urn:ogc:def:crs:EPSG:6.9:4258']  # [UTM, Krueger, WGS84]

epsg_UTM = 25832
epgs_WGS84 = 4258
epsg_Krueger = 31468
