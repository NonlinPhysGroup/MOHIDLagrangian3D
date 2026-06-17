# Created on Thursday 10 April 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

import os
import sys
import numpy as np
from pathlib import Path
import glob


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

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)