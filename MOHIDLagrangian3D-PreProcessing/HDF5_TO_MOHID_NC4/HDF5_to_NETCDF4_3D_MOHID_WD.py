# Created on Thursday 10 April 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

import sys
import h5py
from datetime import datetime
import xarray as xr
import numpy as np
import os
from pathlib import Path
import json
import glob
from netCDF4 import Dataset, date2num
import time as time_package
from colorama import init
import pandas as pd

from HDF5_to_NETCDF4_3D_MOHID_AX import convert_to_int32, convert_to_float32

def HDF5_to_NETCDF4_3D_MOHID_WD(file_path_hdf5_WD, output_data_NETCDF4_foldername):

    '''
        input:  param file_path_hdf5_WD: input file including the wind data
        return: NETCDF4 format compatible with MOHID for 3D simulation
    '''

    print("\033[92m Start conversion from HDF5 to NEtCDF_3D_MOHID[Wind]...\033[0m")

    value = round(-9998.999999, 1)
    fill_val = np.float32(value)

    # Open file
    ds = xr.open_dataset(file_path_hdf5_WD)
    ds = ds.isel(time=slice(0, 24))

    vars_to_keep = ['lon', 'lat', 'time', 'u', 'v']

    # Keep variables + their coordinates + attributes
    subset = ds[vars_to_keep]

    # Make 1D coordinates
    lat_1d = subset['lat'].mean(dim="x")  # average across columns
    lon_1d = subset['lon'].mean(dim="y")  # average across rows average across rows

    # Replace old 2D coords with new 1D ones
    subset = subset.drop_vars(['lat', 'lon'])
    subset = subset.assign_coords(lat=lat_1d, lon=lon_1d)

    # Rename dimensions y->lat, x->lon
    subset = subset.swap_dims({"y": "lat", "x": "lon"})

    subset = subset.rename({'u': 'wind velocity X'})
    subset = subset.rename({'v': 'wind velocity Y'})
    #subset = subset.rename({'u': 'u10'})
    #subset = subset.rename({'v': 'v10'})


    # In the subset coordinate still exist 'x' and 'y' data, we have to remove them:
    subset = subset.drop_vars(['x', 'y'], errors="ignore")
    #subset = subset.reset_coords(names=['x', 'y'], drop=True)

    var_list_all = ['time', 'lat', 'lon', 'wind velocity X', 'wind velocity Y']
    #var_list_all = ['time', 'lat', 'lon', 'u10', 'v10']
    #encoding = {var: {'_FillValue': fill_val} for var in var_list_all}
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

    filename_WD = os.path.splitext(os.path.basename(file_path_hdf5_WD))[0]
    parts1 = filename_WD.split('_')
    parts2 = filename_WD.split('_')
    common_parts = [p for p in parts1 if p in parts2]
    common_string = "_".join(common_parts)
    formatted_time0 = pd.to_datetime(subset['time'].values[0]).strftime('%Y%m%d')
    outputname =common_string + '_[actual_time=' +formatted_time0 + '].nc4'

    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    dir_output = str(exe_dir / output_data_NETCDF4_foldername / 'winds')

    Path(dir_output).mkdir(parents=True, exist_ok=True)
    output_path = Path(dir_output)/outputname
    print('\033[93m Saving NetCDF4 file ...\033[0m')
    subset.to_netcdf(output_path, encoding=encoding, format='NETCDF4')

    print('\033[93m Add variables_ncattrs to NetCDF4 file ...\033[0m')
    variables_ncattrs = ['wind velocity X', 'wind velocity Y']

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

    return