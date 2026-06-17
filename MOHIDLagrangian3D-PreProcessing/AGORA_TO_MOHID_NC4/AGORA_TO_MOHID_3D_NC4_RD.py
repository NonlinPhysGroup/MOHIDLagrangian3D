# Created on Thursday 10 April 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

import os
import numpy as np
from pathlib import Path
import glob

def AGORA_TO_MOHID_3D_NC4_RD(input_data_ROMS_foldername):


    print("\033[94m Reading ROMS files...\033[0m")
    if ((Path(input_data_ROMS_foldername)).exists()):
        file_path_ROMS_0_list = sorted(glob.glob(str(Path(input_data_ROMS_foldername)  / '*.nc')))
        if not file_path_ROMS_0_list:
            print(f"\033[91m Input folder is empty\033[0m")
    else:
        file_path_ROMS_0_list = None
        print(f"\033[91m Input folder does not exist\033[0m")

    return  file_path_ROMS_0_list, file_path_ROMS_0_list