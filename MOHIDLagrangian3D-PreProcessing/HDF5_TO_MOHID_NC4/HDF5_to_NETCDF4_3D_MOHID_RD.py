# Created on Thursday 10 April 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

import os
import numpy as np
from pathlib import Path
import glob

def HDF5_to_NETCDF4_3D_MOHID_RD(input_data_HDF5_foldername):

    print("\033[94m Reading HDF5 hydrodynamic files...\033[0m")
    if ((Path(input_data_HDF5_foldername)  / 'hydrodynamic').exists()):
        file_path_hdf5_HD_list = sorted(glob.glob(str(Path(input_data_HDF5_foldername)  / 'hydrodynamic'    /'*.hdf5')))
        if not file_path_hdf5_HD_list:
            print(f"\033[91m HDF5 hydrodynamic folder is empty\033[0m")
    else:
        file_path_hdf5_HD_list = None
        print(f"\033[91m HDF5 hydrodynamic folder does not exist\033[0m")

    print("\033[94m Reading HDF5 waterProperties files...\033[0m")
    if ((Path(input_data_HDF5_foldername)  / 'waterProperties').exists()):
        file_path_hdf5_WP_list = sorted(glob.glob(str(Path(input_data_HDF5_foldername)  / 'waterProperties'  /'*.hdf5')))
        if not file_path_hdf5_WP_list:
            print(f"\033[91m HDF5 waterProperties folder is empty\033[0m")
    else:
        file_path_hdf5_WP_list = None
        print(f"\033[91m HDF5 waterProperties folder does not exist\033[0m")

    print("\033[94m Reading HDF5 winds files...\033[0m")
    if ((Path(input_data_HDF5_foldername)  / 'winds').exists()):
        file_path_hdf5_WD_list = sorted(glob.glob(str(Path(input_data_HDF5_foldername)  / 'winds'  /'*.nc4')))
        if not file_path_hdf5_WD_list:
            print(f"\033[91m HDF5 winds folder is empty\033[0m")
    else:
        file_path_hdf5_WD_list = None
        print(f"\033[91m HDF5 winds folder does not exist\033[0m")

    print("\033[94m Reading HDF5 wavs files...\033[0m")
    if ((Path(input_data_HDF5_foldername) / 'waves').exists()):
        file_path_hdf5_WV_list = sorted(glob.glob(str(Path(input_data_HDF5_foldername) / 'waves' / '*.nc4')))
        if not file_path_hdf5_WV_list:
            print(f"\033[91m HDF5 waves folder is empty\033[0m")
    else:
        file_path_hdf5_WV_list = None
        print(f"\033[91m HDF5 waves folder does not exist\033[0m")

    return file_path_hdf5_HD_list, file_path_hdf5_WP_list, file_path_hdf5_WD_list,file_path_hdf5_WV_list
