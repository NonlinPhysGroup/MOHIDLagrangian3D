# Created on Thursday 10 April 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

import os
import numpy as np
from pathlib import Path
import glob
from netCDF4 import Dataset, date2num
from datetime import datetime


# Function to convert values to 32-bit float
def convert_to_float32(attributes_dict):
    for key, value in attributes_dict.items():
        if isinstance(value, (int, float)):  # Ensure it’s a number before conversion
            attributes_dict[key] = np.float32(value)
    return attributes_dict

def convert_to_int32(attributes_dict):
    for key, value in attributes_dict.items():
        if isinstance(value, (int, float)):  # Ensure it’s a number before conversion
            attributes_dict[key] = np.int32(value)
    return attributes_dict


def add_variables_ncattrs_to_netcdf4(output_path, subset, variables_ncattrs, formatted_time0):

    print('\033[93m Add variables_ncattrs to NetCDF4 file ...\033[0m')

    history_value = str()
    history_append_value = str()
    for var in variables_ncattrs:
        var_file_path = str(output_path).replace('.nc4', f'_{var}.nc4')
        current_time = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
        strin_temp_history = f"{current_time}: ncks -A -v {var} {var_file_path} {output_path}"
        string_temp_history = strin_temp_history.replace("\\", "/")
        history_value = string_temp_history + history_value

        strin_temp_history_append = f"{current_time}: Appended file {var_file_path} had following \"history\" attribute "
        string_temp_history_append  = strin_temp_history_append.replace("\\", "/")
        history_append_value = string_temp_history_append  + history_append_value

        global_attributes = {
            'history': history_value,
            'history_of_appended_files': history_append_value,
            'NCO': formatted_time0
        }

    with Dataset(output_path, 'a') as nc:
        for key, value in global_attributes.items():
            nc.setncattr(key, value)
    subset.close()

    print("\033[92m Conversion complete: file saved in \033[0m", output_path)


def config_encoding_nc4(fill_val, var_list_all):

    encoding = {
        var: {
            '_FillValue': np.int32(fill_val) if var == 'mask' else (np.float32(-fill_val) if var == 'depth' else np.float32(fill_val)),
            'dtype': 'int32' if var == 'mask' else 'float32',
            'zlib': True,           # Enable zlib compression for each variable
            'complevel': 3,         # Compression level (1-9, where 9 is highest compression)
            'shuffle': True,        # Shuffle filter for better compression
        }
        for var in var_list_all                 # Loop over all variables in the dataset
    }
    return encoding


def wavelength_from_period_depth(T, h, max_iter=20):
    """
    Compute wavelength (lambda) from wave period (T) and depth (h)
    using the linear dispersion relation (ω² = g k tanh(kh)).

    T: wave period (s)
    h: water depth (m)
    Returns: wavelength (m)
    """
    g = 9.81516
    omega = 2 * np.pi / T  # angular frequency

    # Initial guess: deep water
    k = omega ** 2 / g

    # Newton-Raphson iteration
    for _ in range(max_iter):
        print(_)
        kh = k * h
        f = g * k * np.tanh(kh) - omega ** 2
        df = g * np.tanh(kh) + g * k * h * (1 / np.cosh(kh)) ** 2
        k = k - f / df

    wavelength = 2 * np.pi / k
    return wavelength



