import pandas as pd
import datetime

####################################################
## This script returns the index of a pandas
## dataframe with specific time value
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 08.2020
####################################################


def get_df_index_to_timestamp(df: pd.DataFrame, col_time: str, time_marg: datetime) -> int:
    """ Returns the index to the nearest timestamp from df to time_marg
    :param df:          dataframe with time column in Timestamp/datetime64 format
    :param col_time:    name of the time column in df
    :param time_marg:   timestamp to search for as type(Timestamp)
    :return: index of df with near timestamp, if not available None
    """
    time_series = abs(df[col_time] - time_marg).dt.seconds

    # define max seconds (here 5) difference between timestamps
    if time_series.min() <= 5:
        idx = time_series.idxmin()
        return idx
    # else
    return None
