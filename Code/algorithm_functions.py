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
        DIF = MIR - TIR

        # Calculate Thresholds
        thresh_1 = np.where(MIR > thresholds['MIR'], 1, 0)
        thresh_2 = np.where(DIF > thresholds['DIF'], 1, 0)
        potential_fire_pixel = np.where((thresh_1+thresh_2) == 2, 1, 0)

        # Save file
        print('\tSaving thresholds')
        np.save(os.path.join(folder, 'initThresh_MODIS'), potential_fire_pixel)

    os.chdir(cwd)
