import pandas as pd
from progressbar import ProgressBar
pbar = ProgressBar()

####################################################
## This script changes the driveway names to "oneway"
## driveways have the same name as road and need to
## be changed for map matching
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################


def change_driveway_names(df_mapdata: pd.DataFrame) -> pd.DataFrame:

    # get indices for one way roads
    bool_same_oneWay = df_mapdata.sameDirection == 0
    bool_against_oneWay = df_mapdata.againstDirection == 0
    bool_both = bool_same_oneWay | bool_against_oneWay

    df_mapdata.roadName.loc[bool_both] = (df_mapdata.roadName.loc[bool_both] + ' oneWay')

    return df_mapdata
