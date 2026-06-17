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

def HDF5_to_NETCDF4_3D_MOHID_WV(file_path_hdf5_WD, output_data_NETCDF4_foldername):

    '''
        input:  param file_path_hdf5_WD: input file including the waves data
        return: NETCDF4 format compatible with MOHID for 3D simulation
    '''

    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    dir_output = str(exe_dir / output_data_NETCDF4_foldername / 'waves')

    print("\033[92m Start conversion from HDF5 to NEtCDF_3D_MOHID[Waves]...\033[0m")
    print("\033[91m The Funciton from HDF5 to NEtCDF_3D_MOHID[Waves] is empty...\033[0m")


    return