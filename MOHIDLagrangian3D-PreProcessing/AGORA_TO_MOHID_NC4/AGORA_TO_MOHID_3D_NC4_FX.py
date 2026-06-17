# Created on Thursday 10 April 2025

# Author: Mohsen Shabani
# E-mail: shabani.mohsen@outlook.com

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
from pandas.core.computation.expressions import where
from dask.diagnostics import ProgressBar
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from AGORA_TO_MOHID_3D_NC4_AX import convert_to_int32, convert_to_float32 , wavelength_from_period_depth
from AGORA_TO_MOHID_3D_NC4_AX import add_variables_ncattrs_to_netcdf4, config_encoding_nc4


def AGORA_TO_MOHID_3D_NC4_0(file_path_ROMS_0, output_data_NETCDF4_foldername,convert):

    '''
        input:  param file_path_ROMS_0: input file forms ROMS
        return: NETCDF4 format compatible with MOHID for 3D simulation
    '''

    print("\033[92m Start conversion from ROMS to NEtCDF_3D_MOHID format...\033[0m")

    value = round(-9999.1, 1)
    fill_val = np.float32(value)

    # Open file
    ds = xr.open_dataset(file_path_ROMS_0)
    if "u" in ds:
        ds['mask'] = xr.where(~np.isnan(ds['u'][0, :, :, :]), 0, 1)
    ds_set = ds.copy(deep=True)

    #ds_set = ds_set.isel(time=slice(0, 5))
    #ds_set = ds_set.isel(longitude=slice(10, 390))
    ds_set = ds_set.isel(latitude=slice(10, 390))

    #ds_set['VTPK']  = ds_set['WLEN']
    if "longitude" in ds_set:
        ds_set = ds_set.rename({'longitude': 'lon'})
    if "latitude" in ds_set:
        ds_set = ds_set.rename({'latitude': 'lat'})
    if "zeta" in ds_set:
        ds_set = ds_set.rename({'zeta': 'water level'})
    if "h" in ds_set:
        ds_set = ds_set.rename({'h': 'bathymetry'})
    if "u_stokes" in ds_set:
        ds_set = ds_set.rename({'u_stokes': 'vsdx'})
    if "v_stokes" in ds_set:
        ds_set = ds_set.rename({'v_stokes': 'vsdy'})
    if "VHM0" in ds_set:
        ds_set = ds_set.rename({'VHM0': 'significant wave height'})
    if "VSMC" in ds_set:
        ds_set = ds_set.rename({'VSMC': 'mean wave period'})
    if "VMDR" in ds_set:
        ds_set = ds_set.rename({'VMDR': 'mean wave direction'})
#    ds_set = ds_set.rename({'VTPK': 'wave period'})
    if "WLEN" in ds_set:
        ds_set = ds_set.rename({'WLEN': 'wave length'})

#    vars_to_keep = ['longitude', 'latitude', 'time', 'u', 'v', 'water level', 'temp', 'salt', 'VHM0', 'VSMC', 'VMDR', 'VTPK']
    vars_to_keep_Coordinates = ['lon', 'lat', 'depth', 'time'] if "depth" in ds_set else ['lon', 'lat', 'time']
    vars_to_keep_HD = [ 'bathymetry','mask','u', 'v', 'w', 'water level'] if convert.get("current", False) else []
    vars_to_keep_WP = [ 'bathymetry','mask', 'temp','salt'] if convert.get("waterProperties", False) else []

    vars_to_keep_WV_OPT1 = [ 'vsdx', 'vsdy']
    vars_to_keep_WV_OPT2 = [ 'significant wave height','mean wave period','mean wave direction', 'wave length'] #, 'water level'
    vars_to_keep_WV_OPT3 = [ 'vsdx', 'vsdy', 'significant wave height','mean wave period','mean wave direction', 'wave length', 'water level']
    vars_to_keep_WV_OPT4 = [ var for var in vars_to_keep_WV_OPT3 if var in ds_set.variables]
    vars_to_keep_WV = vars_to_keep_WV_OPT4 if convert.get("wave", False) else []

    # Keep variables + their coordinates + attributes
    subset_HD = xr.merge([ ds_set[vars_to_keep_Coordinates], ds_set[vars_to_keep_HD] ])
    subset_WP = xr.merge([ ds_set[vars_to_keep_Coordinates], ds_set[vars_to_keep_WP] ])
    subset_WV = xr.merge([ ds_set[vars_to_keep_Coordinates], ds_set[vars_to_keep_WV] ])

    if "depth" in vars_to_keep_Coordinates:
        subset_HD = subset_HD.sortby("depth", ascending=False)
        subset_WP = subset_WP.sortby("depth", ascending=False)
        subset_WV = subset_WV.sortby("depth", ascending=False)

        #In MOHIDLagrangian, it just take stokes drift at the surface. Therefore, if you give as a function of depth, it fails.
        subset_WV = subset_WV.isel(depth=-1).drop_vars("depth")

    ## Make 1D coordinates
    #lat_1d = subset_HD['lat'].mean(dim="x")  # average across columns
    #lon_1d = subset_HD['lon'].mean(dim="y")  # average across rows average across rows

    ## Replace old 2D coords with new 1D ones
    #subset_HD = subset_HD.drop_vars(['lat', 'lon'])
    #subset_HD = subset_HD.assign_coords(lat=lat_1d, lon=lon_1d)

    # Rename dimensions y->lat, x->lon
    #subset_HD = subset_HD.swap_dims({"y": "lat", "x": "lon"})

    # Add new variable to subset_HD
    if convert.get("current", False):
        subset_HD['water column'] = subset_HD['water level'] + subset_HD['bathymetry'].broadcast_like(subset_HD['water level'])
#        subset_HD['ssh_netcdf'] = subset_HD['water level']
#        subset_HD['sshnetcdf'] = subset_HD['water level']

    filename_ROMS_0 = os.path.splitext(os.path.basename(file_path_ROMS_0))[0]
    parts1 = filename_ROMS_0.split('_')
    parts2 = filename_ROMS_0.split('_')
    common_parts = [p for p in parts1 if p in parts2]
    common_string = "_".join(common_parts)
    formatted_time0 = pd.to_datetime(subset_HD['time'].values[0]).strftime('%Y%m%d')

    if convert.get("current", False):
        outputname_HD = 'Hydrodynamic_' +common_string + '_[actual_time=' +formatted_time0 + '].nc4'
        dir_output_HD = str(Path(__file__).resolve().parent /output_data_NETCDF4_foldername/ 'hydrodynamic')
        Path(dir_output_HD).mkdir(parents=True, exist_ok=True)
        output_path_HD = Path(dir_output_HD)/outputname_HD
        print('\033[93m Saving Hydrodynamic NetCDF4 file ...\033[0m')

        subset_HD = subset_HD.chunk({"time":24})
        with ProgressBar():
            print(subset_HD.chunks)
            subset_HD.to_netcdf(output_path_HD, encoding=config_encoding_nc4(fill_val, list(subset_HD.coords) + list(subset_HD.data_vars)), format='NETCDF4')
        print('\033[93m Add ncattrs Hydrodynamic NetCDF4 file ...\033[0m')
        add_variables_ncattrs_to_netcdf4(output_path_HD, subset_HD, vars_to_keep_HD, formatted_time0)

    if convert.get("waterProperties", False):
        mvd = "mean wave direction"
        if mvd in subset_WV:
            data = subset_WV[mvd].values
            subset_WV[mvd].values[:] = np.where(np.isnan(data),fill_val, (270.0 - data) % 360.0)
        outputname_WP = 'WaterProperties_' +common_string + '_[actual_time=' +formatted_time0 + '].nc4'
        dir_output_WP = str(Path(__file__).resolve().parent /output_data_NETCDF4_foldername/ 'waterProperties')
        Path(dir_output_WP).mkdir(parents=True, exist_ok=True)
        output_path_WP = Path(dir_output_WP)/outputname_WP
        print('\033[93m Saving WaterProperties NetCDF4 file ...\033[0m')
        subset_WP = subset_WP.chunk({"time":24})
        with ProgressBar():
            subset_WP.to_netcdf(output_path_WP, encoding=config_encoding_nc4(fill_val, list(subset_WP.coords) + list(subset_WP.data_vars)), format='NETCDF4')
        print('\033[93m Add ncattrs  WaterProperties NetCDF4 file ...\033[0m')
        add_variables_ncattrs_to_netcdf4(output_path_WP, subset_WP, vars_to_keep_WP, formatted_time0)

    if convert.get("wave", False):
        outputname_WV = 'Waves_' +common_string + '_[actual_time=' +formatted_time0 + '].nc4'
        dir_output_WV = str(Path(__file__).resolve().parent /output_data_NETCDF4_foldername/ 'waves')
        Path(dir_output_WV).mkdir(parents=True, exist_ok=True)
        output_path_WV = Path(dir_output_WV)/outputname_WV
        print('\033[93m Saving Waves NetCDF4 file ...\033[0m')
        subset_WV = subset_WV.chunk({"time":24})
        with ProgressBar():
            subset_WV.to_netcdf(output_path_WV, encoding=config_encoding_nc4(fill_val, list(subset_WV.coords) + list(subset_WV.data_vars)), format='NETCDF4')
        print('\033[93m Add ncattrs  Waves NetCDF4 file ...\033[0m')
        add_variables_ncattrs_to_netcdf4(output_path_WV, subset_WV, vars_to_keep_WV, formatted_time0)

    return