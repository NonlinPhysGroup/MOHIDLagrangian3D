# Created on Thursday, 11 November 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

import os
import sys
import numpy as np
from pathlib import Path
import glob
import json
from HDF5_to_NETCDF4_3D_MOHID_RD import HDF5_to_NETCDF4_3D_MOHID_RD
from HDF5_to_NETCDF4_3D_MOHID_HD import HDF5_to_NETCDF4_3D_MOHID_HD
from HDF5_to_NETCDF4_3D_MOHID_WP import HDF5_to_NETCDF4_3D_MOHID_WP
from HDF5_to_NETCDF4_3D_MOHID_WD import HDF5_to_NETCDF4_3D_MOHID_WD
from HDF5_to_NETCDF4_3D_MOHID_WV import HDF5_to_NETCDF4_3D_MOHID_WV
from HDF5_to_NETCDF4_3D_MOHID_AX import resource_path

import rasterio
base = os.path.dirname(rasterio.__file__)
# e.g., look for a folder named "gdal_data" under base
candidate = os.path.join(base, "gdal_data")
if os.path.isdir(candidate):
    gdal_data = candidate
    print('gdal_data:', gdal_data)
else:
    print('Not a directory gdal_data' )

if hasattr(sys, "_MEIPASS"):
    # Running as EXE
    os.environ["GDAL_DATA"] = os.path.join(sys._MEIPASS, "gdal_data")
else:
    # Running as .py script
    import rasterio
    base = os.path.dirname(rasterio.__file__)
    candidate = os.path.join(base, "gdal_data")
    if os.path.isdir(candidate):
        os.environ["GDAL_DATA"] = candidate

print("GDAL_DATA =", os.environ["GDAL_DATA"])

exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(".")

#Do not use resource_path() for external, editable files.

#json_path = resource_path("input_options.json")
json_path = exe_dir / "input_options.json"
with open(json_path, "r") as f:
    params = json.load(f)


# Extract values
#input_data_HDF5_foldername = resource_path(params["input_data_HDF5_foldername"])
#output_data_NETCDF4_foldername = resource_path(params["output_data_NETCDF4_foldername"])
input_data_HDF5_foldername = (params["input_data_HDF5_foldername"])
output_data_NETCDF4_foldername = (params["output_data_NETCDF4_foldername"])
rogusity_option_dict = params.get("rogusity_option_dict", None)

file_path_hdf5_list = []
file_path_hdf5_list = HDF5_to_NETCDF4_3D_MOHID_RD(input_data_HDF5_foldername)

file_path_hdf5_HD_list = file_path_hdf5_list[0]
file_path_hdf5_WP_list = file_path_hdf5_list[1]
file_path_hdf5_WD_list = file_path_hdf5_list[2]
file_path_hdf5_WV_list = file_path_hdf5_list[3]
roughness_option_dict = dict()


if file_path_hdf5_HD_list:
    for k, file_HD in enumerate(file_path_hdf5_HD_list, start=1):
        print(f"\033[92m Files number: {k}\033[0m")
        HDF5_to_NETCDF4_3D_MOHID_HD(file_HD, output_data_NETCDF4_foldername, rogusity_option_dict)

if file_path_hdf5_WP_list:
    for k, file_WP in enumerate(file_path_hdf5_WP_list, start=1):
        print(f"\033[92m Files number: {k}\033[0m")
        HDF5_to_NETCDF4_3D_MOHID_WP(file_WP, output_data_NETCDF4_foldername)

if file_path_hdf5_WD_list:
    for k, file_WD in enumerate(file_path_hdf5_WD_list, start=1):
        print(f"\033[92m Files number: {k}\033[0m")
        HDF5_to_NETCDF4_3D_MOHID_WD(file_WD, output_data_NETCDF4_foldername)

if file_path_hdf5_WV_list:
    for k, file_WV in enumerate(file_path_hdf5_WV_list, start=1):
        print(f"\033[92m Files number: {k}\033[0m")
        HDF5_to_NETCDF4_3D_MOHID_WV(file_WV, output_data_NETCDF4_foldername)



