import os
import pandas as pd
from pathlib import Path


def get_path(dataset=None, extension=None):
    '''
    Returns subfolder containing SCRIPT data.

    Input:
        dataset     str, name of dataset, e.g.: 'received_limited'
        extension   str, optional, subfolder
    Output:
        path        str, folder
    '''

    if extension is not None:
        extension = _adjust_to_current_file_separator(extension)
    
    settings = _settings_from_file()
    
        
    ref_folder = settings[dataset]

    if extension is not None:
        path = os.path.join(ref_folder, extension)
    else:
        path = ref_folder
   
    return path


def _settings_from_file():
    """
    Loads settings from settings file.
    
    Output:
    settings       dict
    """

    home = str(Path.home())

    path_to_settings = os.path.join(
        str(Path.home()), 'Documents', 'data_paths')

    if not os.path.exists(path_to_settings):
        raise EnvironmentError(rf"""
            Could not find directory reserved for settings:
            {path_to_settings}
        """)

    path_to_settings = os.path.join(
        path_to_settings,
        'U19access.csv'
    )

    if not os.path.exists(path_to_settings):
        raise EnvironmentError(rf"""
            Could not find U19access.csv file:
            {path_to_settings}

            This file needs to be UTF-8 formatted
            csv file with two columns: key, value

            Also see readme of repository for
            further guidance.
        """)
        
        

    settings = pd.read_csv(
        path_to_settings
    )
    
    
    if not all(settings.columns == ['key', 'value']):
        raise EnvironmentError(rf"""
            U19access.csv must have exactly two different
            columns, namely key and value.
            
            {path_to_settings}
        """)
        
        
    settings = settings.drop_duplicates()
    
    if any(settings['key'].duplicated()):
        raise EnvironmentError(rf"""
            At least one key within U19access.csv is
            duplicated and therefore ambiguous
            
            {path_to_settings}
        """)
    
    
    settings = settings.set_index(
        'key', 
        verify_integrity=True
    )['value'].to_dict()
    
    return settings


def _adjust_to_current_file_separator(x):
    '''
    Replaces backslashes and forward slashes
    by file separtor used on current operating
    system.
    '''
    x = x.replace('\\', os.path.sep)
    x = x.replace('/', os.path.sep)

    return x