"""
Functions to display and save results
"""
import os
import cartopy.crs as ccrs
import cartopy
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
Display Sentinel Data
"""


def displayData(sentinelsat_options: dict):

    cwd = os.getcwd()
    os.chdir('s3_data')

    path_to_folders = glob.glob('*.SEN3')

    for folder in path_to_folders:
        print('')
        print(folder)

        # Get NADIR View
        # File name (include extenstion) -> and move into this
        file = glob.glob(folder+os.sep+'F1_BT_fn.nc')[0]

        with Dataset(file) as src:
            # Each datafile contains multiple bands
            for band, variable in src.variables.items():
                print('\n------' + band + '------')
                if "_BT_fn" in band:
                    bandName = band
                    for attrname in variable.ncattrs():
                        print("{} -- {}".format(attrname, getattr(variable, attrname)))
        data = Dataset(file)
        data.set_auto_mask(False)
        bandData = data.variables[bandName][:]
        data.close()

        # Get Coordinates of NADIR View
        file = glob.glob(folder+os.sep+'geodetic_fn.nc')[0]

        data = Dataset(file)
        data.set_auto_mask(False)
        lat = data.variables['latitude_fn'][:]
        lon = data.variables['longitude_fn'][:]
        data.close()

        # Plot figures
        lon_min = 23
        lon_max = 24
        lat_min = 38.5
        lat_max = 39.5

        lon_ratio = lon_max-lon_min
        lat_ratio = lat_max-lat_min

        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND)

        # plt.imshow(bandData, vmin=220, vmax=400)
        plt.scatter(lon, lat, s=10, c=bandData,
                    transform=ccrs.PlateCarree(), vmin=220, vmax=400)

        ax.coastlines()
        ax.gridlines(draw_labels=True)

        plt.colorbar()
        plt.show()

    os.chdir(cwd)


"""
Display MIR Band with detected fire pixels
"""


def displayFirePixels(data_dir='s3_data', coordinates={'lon_min': 22, 'lon_max': 24, 'lat_min': 38, 'lat_max': 40}, show=False):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')
    print('Available Folders:')
    for folder in path_to_folders:
        print(folder)

    # Go into each folder and plot
    print('\nDisplaying Initial MODIS Thresholds for Products:')
    for folder in path_to_folders:

        # Get values
        MIR = np.load(os.path.join(folder, 'F1_BT_fn.npy'))
        TIR = np.load(os.path.join(folder, 'F2_BT_in.npy'))
        DIF = MIR - TIR
        potential_fire_pixel = np.load(os.path.join(folder, 'initThresh_MODIS.npy'))
        lat = np.load(os.path.join(folder, 'latitude_fn.npy'))
        lon = np.load(os.path.join(folder, 'longitude_fn.npy'))

        # Make Figure
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([coordinates['lon_min'], coordinates['lon_max'],
                      coordinates['lat_min'], coordinates['lat_max']], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND)
        ax.gridlines(draw_labels=True)

        # Add MIR Band
        cmap = cm.get_cmap('Spectral_r')
        # plt.imshow(MIR, vmin=220, vmax=400, cmap=cmap)
        plt.scatter(lon, lat, s=30, c=MIR, transform=ccrs.PlateCarree(),
                    vmin=220, vmax=400, cmap=cmap, alpha=1)
        plt.colorbar()

        # Add Fire Pixels
        cmap = cm.get_cmap('Reds', 2)
        plt.scatter(lon, lat, s=1, c=potential_fire_pixel,
                    transform=ccrs.PlateCarree(), vmin=0, vmax=1, alpha=1, cmap=cmap)

        # Create title and show
        plt.title("F1 Band with Potential Fire Pixels Overlaid in Red")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        if show:
            plt.show()


"""
Display MIR Bands for Cells
"""


def displayCellValue(grid_info: dict, data_dir='s3_data'):

    id_map = grid_info['id_map']
    x_center = id_map[:, -2]
    y_center = id_map[:, -1]
    print()
    print(y_center)
    print(x_center)

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')

    # Go into each folder and plot
    print('\nDisplaying Initial MODIS Thresholds for Products:')
    for folder in path_to_folders:

        pixel_cell_info = np.load(os.path.join(folder, 'pixel_cell_info.npy'))
        cell_value = np.load(os.path.join(folder, 'cell_value.npy'))

        # print(cell_value[:, 1])
        # print(x_center.shape)
        # print(y_center.shape)

        # Plot image
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND)
        cmap = cm.get_cmap('Reds', 2)
        plt.scatter(x_center, y_center, s=90,
                    c=cell_value[:, 1], transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.5, marker='s')
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        # plt.colorbar()
        plt.title("Potential Fire Areas (Resolution: 0.05 Degrees)")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.show()
