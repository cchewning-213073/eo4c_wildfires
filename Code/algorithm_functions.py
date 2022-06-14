"""
Functions for Active Fire Detection
"""
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import glob
from netCDF4 import Dataset
import zipfile
import cartopy.crs as ccrs
from pylab import cm
import geopandas as gpd

from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt


"""
Compute Initial Thresholds:
    MIR, TIR, diff MIR, zenith -  Sentinel Data
"""


def computeInitThresh_MODIS(data_dir='s3_data', thresholds={'MIR': 310, 'DIF': 10}):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')
    print('Available Folders:')
    for folder in path_to_folders:
        print(folder)

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Initial MODIS Thresholds for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1

        # Get values
        MIR = np.load(os.path.join(folder, 'F1_BT_fn.npy'))
        TIR = np.load(os.path.join(folder, 'F2_BT_in.npy'))
        DIF = np.load(os.path.join(folder, 'DIF.npy'))

        # Calculate Thresholds
        thresh_1 = np.where(MIR > thresholds['MIR'], 1, 0)
        thresh_2 = np.where(DIF > thresholds['DIF'], 1, 0)
        potential_fire_pixel = np.where((thresh_1+thresh_2) == 2, 1, 0)

        # Save file
        print('\tSaving thresholds')
        np.save(os.path.join(folder, 'DIF'), DIF)
        np.save(os.path.join(folder, 'initThresh_MODIS'), potential_fire_pixel)

    os.chdir(cwd)


"""
Compute Initial Thresholds:
    MIR, TIR, diff MIR, zenith -  Sentinel Data
"""


def createGrid(res=0.05, coordinates={'lon_min': 22, 'lon_max': 24, 'lat_min': 38, 'lat_max': 40}):

    # Define the values that indicate the grid lines
    x_bounds = np.arange(coordinates['lon_min'], coordinates['lon_max'], res)
    y_bounds = np.arange(coordinates['lat_min'], coordinates['lat_max'], res)

    # Define the values that mark the center coordinates of the grid
    x_center = x_bounds[:-1]+(0.5*res)
    y_center = y_bounds[:-1]+(0.5*res)

    # Create Point Map: then size of the meshed center values, but we label them with an indexes
    # make grid
    id = np.reshape(np.arange(len(x_center)*len(y_center)), (len(y_center), len(x_center)))

    # Wrap it all up for the output
    grid = {
        'x_bounds': x_bounds,
        'y_bounds': y_bounds,
        'x_center': x_center,
        'y_center': y_center,
        'id': id,
    }

    return grid
