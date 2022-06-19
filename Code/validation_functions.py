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
import skimage.util as ski


def extractValidationData(data_dir='s3_FRP_data'):

	# load data

	# Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and calculate threshold for each product
    print('\nExtracting Validation Data for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tProcessing product: {count}/{len(path_to_folders)}')

        flags = np.load(os.path.join(folder, 'flags.npy'))

        print(flags[0,0])


    os.chdir(cwd)