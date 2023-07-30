########################################################
## Change pandas datatype to float or int from string
## Output nan if convertion failed e.g. for empty string
########################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
########################################################
## Date: 05.2020
########################################################
## Example
## to_float = ['angel', 'sectionLength', 'station']
## to_int = ['fromNode', 'toNode', 'sectionNumber', 'abschnittslaenge']
## df_Hatches = pandas_to_numeric(df_Hatches, to_int, to_float)
########################################################

import pandas as pd


def pandas_to_numeric(df_input, to_int=[], to_float=[], conv_index=True):
    for n in to_int:
        df_input[n] = pd.to_numeric(df_input[n], errors='coerce', downcast='integer')
    for n in to_float:
        df_input[n] = pd.to_numeric(df_input[n], errors='coerce')
    if conv_index:
        df_input.index = pd.to_numeric(df_input.index, errors='coerce', downcast='integer')

    return df_input
