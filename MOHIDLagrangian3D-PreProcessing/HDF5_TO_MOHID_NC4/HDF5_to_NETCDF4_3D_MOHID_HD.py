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
from QGIS_to_NETCDF4_RG import process_shapefiles_and_rugosity


def HDF5_to_NETCDF4_3D_MOHID_HD(file_path_hdf5_HD, output_data_NETCDF4_foldername, rogusity_option_dict=None):


    '''
        input:  param file_path_hdf5_HD: input file including the HydroDynamic data
        return: NETCDF4 format compatible with MOHID for 3D simulation
    '''

    print("\033[92m Start conversion from HDF5 to NEtCDF_3D_MOHID [Hydrodynamic]...\033[0m")

    value = round(-9998.999999, 1)
    fill_val = np.float32(value)

    rootGridGroup = '/Grid/'
    rootResultsGroup = '/Results/'
    rootTimeGroup = '/Time/'
    # nameTimesNameGroup = '/Time/Time_'

    HD_hdf5_file = h5py.File(file_path_hdf5_HD, 'r')

    latitudeIn = HD_hdf5_file[rootGridGroup + '/Latitude']
    longitudeIn = HD_hdf5_file[rootGridGroup + '/Longitude']
    rank = len(latitudeIn.shape)
    lat = latitudeIn[0, :] if rank == 2 else latitudeIn[:]
    lon = longitudeIn[:, 0] if rank == 2 else longitudeIn[:]

    bathymetryIn = HD_hdf5_file[rootGridGroup + '/Bathymetry']
    bathymetry = np.where(bathymetryIn[:,:] > 0, bathymetryIn[:,:], fill_val)
    #bathymetry = bathymetryIn[:,:]
    waterpoints3DIn = HD_hdf5_file[rootGridGroup + '/WaterPoints3D']
    waterpoints3D = 1 - waterpoints3DIn[:, :, :]

    bathymetry =  np.where(waterpoints3D[-1,:,:] == 1, fill_val, bathymetry )

    timesGroup = HD_hdf5_file[rootTimeGroup]
    time_t = []
    time_t_second = []
    time_t_second_2 =[]
    z_t = []
    u_t, v_t, w_t = [], [], []
    wl_t, wc_t = [], []

    for i, nameTime in enumerate(timesGroup):
        if i == 24:
            break

        numName = nameTime.split('_')[1]
        # valNumName = int(numName)
        rootNameTime = rootTimeGroup + nameTime
        time = HD_hdf5_file[rootNameTime]

        timeIn = datetime(int(time[0]), int(time[1]), int(time[2]), int(time[3]), int(time[4]), int(time[5]))
        # nameData = '%04d/%02d/%02d %02d:%02d:%02d' % (int(time[0]),int(time[1]),int(time[2]),int(time[3]),int(time[4]),int(time[5]))

        nameZ = rootGridGroup + '/VerticalZ/Vertical_' + numName
        nameU = rootResultsGroup + "velocity U/velocity U_" + numName
        nameV = rootResultsGroup + "velocity V/velocity V_" + numName
        nameW = rootResultsGroup + "velocity W/velocity W_" + numName
        nameWL = rootResultsGroup + "water level/water level_" + numName
        nameWC = rootResultsGroup + "water column/water column_" + numName

        z = HD_hdf5_file[nameZ]
        u = HD_hdf5_file[nameU]
        v = HD_hdf5_file[nameV]
        w = HD_hdf5_file[nameW]
        wl= HD_hdf5_file[nameWL]
        wc= HD_hdf5_file[nameWC]

        time_t.append(timeIn)
        dt_1 = time_t[0]
        dt_2 = timeIn
        dt   = dt_2 - dt_1
        time_t_second.append(dt.total_seconds())

        time_units = 'seconds since ' + time_t[0].strftime('%Y-%m-%d %H:%M:%S')
        calendar = "standard"
        time_t_second_2.append(date2num(timeIn, units=time_units, calendar=calendar))

        z_t.append(z[:, :, :])
        u_t.append(u[:, :, :])
        v_t.append(v[:, :, :])
        w_t.append(w[:, :, :])
        wl_t.append(wl[:, :])
        wc_t.append(wc[:, :])

    depth = np.stack(z_t)
    U, V, W = map(np.stack, (u_t, v_t, w_t))
    WL, WC= map(np.stack, (wl_t, wc_t))

    lon_c = (lon[1:]+lon[0:-1])/2.
    lat_c = (lat[1:]+lat[0:-1])/2.
    depth_c = (depth[:, 1:, :, :]+depth[:, 0:-1, :, :])/2.
    depth_c_avg = []

    valid_bathy = bathymetry[bathymetry >= 0]
    bathy_min = np.min(valid_bathy)

    for i in range(waterpoints3D.shape[0]):
        print(f" \033[93m \r Calculating average depth in nivel {i}\033[0m", end='', flush=True)
        #time_package.sleep(0.5)
        #values = [depth_c[0, i, j, k].item() for j, k in np.ndindex(waterpoints3D.shape[1], waterpoints3D.shape[2]) if waterpoints3D[i, j, k] == 0]
        #values = [depth_c[0, i, j, k].item() for j, k in np.ndindex(waterpoints3D.shape[1], waterpoints3D.shape[2]) if depth_c[0, i, j, k].item()>0]
        #values = [depth_c[0, i, j, k].item() for j, k in np.ndindex(waterpoints3D.shape[1], waterpoints3D.shape[2]) if (waterpoints3D[i, j, k] == 0 and depth_c[0, i, j, k].item()>0)]

        top_depth = 0.001
        values = [(depth_c[0, i, j, k].item() -depth_c[0, waterpoints3D.shape[0] - 1, j, k].item() if i != (waterpoints3D.shape[0]-1) else top_depth) for j, k in np.ndindex(waterpoints3D.shape[1], waterpoints3D.shape[2]) if (waterpoints3D[i, j, k].item() == 0)]

        depth_c_avg.append((np.mean(values).item()) if values else -fill_val)

    # depth_c : depth_c['time', 'depth', 'lon', 'lat']
    # U : U['time', 'depth', 'lon', 'lat']
    # V : V['time', 'depth', 'lon', 'lat']
    # W : W['time', 'depth', 'lon', 'lat']
    # WL : WL['time', 'lon', 'lat']
    # temp : temp['time', 'depth', 'lon', 'lat']
    # salt : salt['time', 'depth', 'lon', 'lat']
    # bathymetry : bathymetry['lon', 'lat']
    # mask : mask['depth', 'lon', 'lat']
    coords = {'lon': (['lon'], lon_c), 'lat': (['lat'], lat_c),'depth': (['depth'], depth_c_avg) , 'time': (['time'], time_t
                                                                                                            )}
    ds = xr.Dataset(coords=coords,
                    data_vars={
                                'bathymetry': (['lat', 'lon'], bathymetry.transpose(1, 0)),
                                'mask': (['depth', 'lat', 'lon'], waterpoints3D.transpose(0, 2, 1)),
                                'u': (['time', 'depth', 'lat', 'lon'], U.transpose(0, 1, 3, 2)),
                                'v': (['time', 'depth', 'lat', 'lon'], V.transpose(0, 1, 3, 2)),
                                'w': (['time', 'depth', 'lat', 'lon'], W.transpose(0, 1, 3, 2)),
                                'water_level': (['time', 'lat', 'lon'], WL.transpose(0, 2, 1)),
                                'water_column': (['time', 'lat', 'lon'], WC.transpose(0, 2, 1)),
                                }
                    )

    # Replace zeroes with fill value
    var_list = [ 'depth', 'lat', 'lon', 'u', 'v', 'w','water_level', 'water_column', 'bathymetry']
    for var in var_list:
        if var == 'depth':
            ds[var] = ds[var].where((ds[var] > -1e15) & (ds[var] != 0.0), -fill_val)
        else:
            ds[var] = ds[var].where((ds[var] > -1e15) & (ds[var] != 0.0), fill_val)
            #ds[var] = ds[var].where((ds[var] > -1e15), fill_val)
    time_atts = {'long_name': 'time'
                 }

    lon_attributtes = {'long_name': 'longitude',
                       'standard_name': 'longitude',
                       'units': 'degrees_east',
                       'valid_min': -180.0,
                       'valid_max': +180.0,
                        'maximum' : np.max(ds['lon']).item(),
                        'minimum' : np.min(ds['lon']).item()}

    lat_attributtes = {'long_name': 'latitude',
                       'standard_name': 'latitude',
                       'units': 'degrees_north',
                       'valid_min': -90.0,
                       'valid_max': +90.0,
                       'maximum' : np.max(ds['lat']).item(),
                       'minimum' : np.min(ds['lat']).item()}

    bath_attributtes = {'long_name' : "bathymetry below minum sea level",
                        'standard_name' : "sea_floor_depth_below_geoid",
                        'units' : "meters",
                        'valid_min' : -50.0,
                        'valid_max' : 11000.0,
                        'maximum' : np.max(ds['bathymetry']).item(),
                        'minimum' : np.min(ds['bathymetry']).item()}
    water_level_attributtes = {'long_name' : "water_level (sea surface height)",
                        'standard_name' : "water level (sea surface height)",
                        'units' : "meters",
                        'valid_min' : -10.0,
                        'valid_max' :+200.0,
                        'maximum' : np.max(ds['water_level']).item(),
                        'minimum' : np.min(ds['water_level']).item()}
    water_column_attributtes = {'long_name' : "water_column",
                        'standard_name' : "water column",
                        'units' : "meters",
                        'valid_min' : -10.0,
                        'valid_max' :+200.0,
                        'maximum' : np.max(ds['water_column']).item(),
                        'minimum' : np.min(ds['water_column']).item()}
    mask_attributtes = {'long_name' : "land_binary_mask",
                        'standard_name' : "land_binary_mask",
                        'units' : "1",
                        'valid_min' : 0,
                        'valid_max' : 1,
                        'maximum' : 0,
                        'minimum' : 1}

    # Convert all attribute dictionaries to float32
    lon_attributtes = convert_to_float32(lon_attributtes)
    lat_attributtes = convert_to_float32(lat_attributtes)
    bath_attributtes = convert_to_float32(bath_attributtes)
    water_level_attributtes = convert_to_float32(water_level_attributtes)
    water_column_attributtes = convert_to_float32(water_column_attributtes)
    mask_attributtes = convert_to_int32(mask_attributtes)

    ds.time.attrs = time_atts
    ds.lon.attrs = lon_attributtes
    ds.lat.attrs = lat_attributtes
    ds.bathymetry.attrs = bath_attributtes
    ds.water_level.attrs = water_level_attributtes
    ds.water_column.attrs = water_column_attributtes
    ds.mask.attrs = mask_attributtes

    #ds.time.encoding['units'] = 'seconds since ' + time_t[0].strftime('%Y-%m-%d %H:%M:%S')
    #ds.time.attrs['units'] = time_units
    #ds.time.attrs['calendar'] = calendar

    var_list_all = ['time', 'depth', 'lat', 'lon', 'u', 'v', 'w','water_level','water_column', 'mask', 'bathymetry']
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

    filename_HD = os.path.splitext(os.path.basename(file_path_hdf5_HD))[0]
    parts1 = filename_HD.split('_')
    parts2 = filename_HD.split('_')
    common_parts = [p for p in parts1 if p in parts2]
    common_string = "_".join(common_parts)
    outputname =common_string + '_[actual_time=' +time_t[0].strftime('%Y%m%d') + '].nc4'

    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    dir_output = str(exe_dir / output_data_NETCDF4_foldername / 'hydrodynamic')

    Path(dir_output).mkdir(parents=True, exist_ok=True)
    output_path = Path(dir_output)/outputname
    print('\n \033[93m Saving NetCDF4 file ...\033[0m')

    # -----------------------------
    # Example usage
    # -----------------------------

    if rogusity_option_dict is not None:
        shapefile_folder = rogusity_option_dict['shapefile_folder']
        value_column = rogusity_option_dict['value_column']
        rugosity_method= rogusity_option_dict['rugosity_method']
        rugosity_map = rogusity_option_dict['rugosity_map']
        use_background = rogusity_option_dict['use_background']
        percent_in_0_1 = rogusity_option_dict['percent_in_0_1']
        background_rugosity = rogusity_option_dict['background_rugosity']
        kmz_folder = rogusity_option_dict['kmz_folder']
        kmz_rugosity_map = rogusity_option_dict['kmz_rugosity_map']
        default_kmz_rugosity = rogusity_option_dict['default_kmz_rugosity']
        land_mask_shp  = rogusity_option_dict['land_mask_shp']
        conversion_rugosity_height_to_length = rogusity_option_dict['conversion_rugosity_height_to_length']

        ds = process_shapefiles_and_rugosity(
            ds=ds,
            shapefile_folder=shapefile_folder,
            value_column=value_column,
            rugosity_method=rugosity_method,
            rugosity_map=rugosity_map,
            use_background=use_background,
            percent_in_0_1=percent_in_0_1,
            background_rugosity=background_rugosity,
            kmz_folder=kmz_folder,
            kmz_rugosity_map=kmz_rugosity_map,
            default_kmz_rugosity=default_kmz_rugosity,
            land_mask_shp=land_mask_shp,
            conversion_rugosity_height_to_length=conversion_rugosity_height_to_length,
            fillValue=fill_val,
            fillValue_2=0
        )
    else:
        print('\033[93m rogusity_option_dict is not available\033[0m')

    # -----------------------------
    # Example usage
    # -----------------------------
    encoding_dict = {
        var: {"_FillValue": fill_val}
        for var in ds.data_vars
        if var not in var_list_all
    }
    combined_encoding = {**encoding_dict, **encoding}

    ds.to_netcdf(output_path, encoding=combined_encoding, format='NETCDF4')

    print('\033[93m Add variables_ncattrs to NetCDF4 file ...\033[0m')
    variables_ncattrs = ['water_level', 'u', 'v', 'w', 'bathymetry']

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
            'NCO': time_t[0].strftime('%Y%m%d')
        }

    with Dataset(output_path, 'a') as nc:
        for key, value in global_attributes.items():
            nc.setncattr(key, value)
    ds.close()

    print("\033[92m Conversion complete: file saved in \033[0m", output_path)

    return