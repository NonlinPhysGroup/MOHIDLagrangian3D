# Created on Thursday 10 April 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

import os
import numpy as np
from pathlib import Path
import glob
import json

from AGORA_TO_MOHID_3D_NC4_RD import AGORA_TO_MOHID_3D_NC4_RD
from AGORA_TO_MOHID_3D_NC4_FX import AGORA_TO_MOHID_3D_NC4_0


with open("input_options.json", "r", encoding="utf-8") as f:
    params = json.load(f)


# Extract values
input_data_ROMS_foldername = Path(params["input_data_ROMS_foldername"])
output_data_NETCDF4_foldername = Path(params["output_data_NETCDF4_foldername"])
convert = dict()
convert = params["convert"]

file_path_ROMS_list = []
file_path_ROMS_list = AGORA_TO_MOHID_3D_NC4_RD(input_data_ROMS_foldername)



file_path_ROMS_0_list = file_path_ROMS_list[0]

if file_path_ROMS_0_list:
    for k, file_0 in enumerate(file_path_ROMS_0_list, start=1):
        print(f"\033[92m Files number: {k}\033[0m {file_0}")
        AGORA_TO_MOHID_3D_NC4_0(file_0, output_data_NETCDF4_foldername, convert)


