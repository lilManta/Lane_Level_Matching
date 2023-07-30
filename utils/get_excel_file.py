import os

####################################################
## This script loads a text file which has
## a folder path saved
####################################################
## Author: Frederic Brenner
## Email: frederic.brenner@tum.de
####################################################
## Date: 06.2020
####################################################

def get_excel_file(excel_file_cursor: str) -> str:
    """ Returns the correct file name according to the input text file

    :param excel_file_cursor: file_name of text file
    :return: folder and file name for excel file
    """
    # open text file
    with open(excel_file_cursor, 'r') as txt_file:
        file_name = txt_file.readline()

    # create absolute file path
    folder_path = os.path.dirname(excel_file_cursor)
    folder_path = folder_path + '/' + file_name
    return folder_path
