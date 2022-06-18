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
import imageio

from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt

"""
Display Sentinel Data
"""
def displayData(sentinelsat_options: dict):

    cwd = os.getcwd()
    os.chdir('s3_data')

    path_to_folders = sorted(glob.glob('*.SEN3'))

    count = 0
    for folder in path_to_folders:
        count += 1
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
        plt.savefig(f'../../figures/gif_files/f1_band/f1_band_{count}.png')
        # plt.show()

    os.chdir(cwd)


"""
Display MIR Band with detected fire pixels
"""
def displayFirePixels(data_dir='s3_data', coordinates={'lon_min': 22, 'lon_max': 24, 'lat_min': 38, 'lat_max': 40}, show=False, confirmed = True):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and plot
    if confirmed: print('\nDisplaying Confirmed Fire Pixels for Products:')
    else:         print('\nDisplaying Potential Fire Pixels for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        if confirmed: print(f'\n\tDisplaying Confirmed Fire Pixels for product {count}/{len(path_to_folders)}')
        else:         print(f'\n\tDisplaying Potential Fire Pixels for product {count}/{len(path_to_folders)}')

        # Get values
        MIR = np.load(os.path.join(folder, 'F1_BT_fn.npy'))
        TIR = np.load(os.path.join(folder, 'F2_BT_in.npy'))
        DIF = MIR - TIR

        if confirmed:
            potential_fire_pixel = np.load(os.path.join(folder, 'confirmed_fire_pixels.npy'))
        else:
            potential_fire_pixel = np.load(os.path.join(folder, 'potential_fire_pixels.npy'))

        lat = np.load(os.path.join(folder, 'latitude_fn.npy'))
        lon = np.load(os.path.join(folder, 'longitude_fn.npy'))

        # Make Figure
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([coordinates['lon_min'], coordinates['lon_max'],
                      coordinates['lat_min'], coordinates['lat_max']], crs=ccrs.PlateCarree())

        # Add MIR Band
        cmap = cm.get_cmap('Spectral_r')
        plt.scatter(lon, lat, s=30, c=MIR, transform=ccrs.PlateCarree(), vmin=220, vmax=400, cmap=cmap, alpha=1)
        plt.colorbar()

        # Add Fire Pixels
        cmap = cm.get_cmap('Reds', 2)
        plt.scatter(lon, lat, s=1, c=potential_fire_pixel,
                    transform=ccrs.PlateCarree(), vmin=0, vmax=1, alpha=1, cmap=cmap)

        ax.coastlines()
        # ax.add_feature(cartopy.feature.OCEAN)
        # ax.add_feature(cartopy.feature.LAND)
        ax.gridlines(draw_labels=True)

        # Create title and show
        if confirmed:
            plt.title(f"F1 Band with Confirmed Fire Pixels: {folder}")
            plt.tight_layout()
            plt.savefig(f'../../../figures/gif_files/confirmed_fire_pixels/confirmed_fire_pixels_{count}.png')
        else:
            plt.title(f"F1 Band with Potential Fire Pixels: {folder}")
            plt.tight_layout()
            plt.savefig(f'../../../figures/gif_files/potential_fire_pixels/potential_fire_pixels_{count}.png')

        if show:
            plt.show()

    os.chdir(cwd)


"""
Display MIR Bands for Cells
"""
def displayCellValue(grid_info: dict, data_dir='s3_data'):

    id_map = grid_info['id_map']
    x_center = id_map[:, -2]
    y_center = id_map[:, -1]

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and plot
    print('\nDisplaying Initial MODIS Thresholds for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1

        pixel_cell_info = np.load(os.path.join(folder, 'pixel_cell_info.npy'))
        cell_value = np.load(os.path.join(folder, 'cell_value.npy'))

        # Save firepixels as matrix
        fire_pixel_output = cell_value[:, 1].reshape((40, 40))
        # np.save(os.path.join(folder, 'active_fires_'+folder), fire_pixel_output)

        fire_folder = os.path.join(cwd, '../active_fire_product')
        np.save(os.path.join(fire_folder, 'active_fires_'+folder), fire_pixel_output)

        # plt.figure(figsize=(8, 8))
        # plt.imshow(fire_pixel_output)
        # plt.show()

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
        plt.title(f"Final Fire Cells (Res: 0.05 Degrees) on {folder}")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'../../../figures/gif_files/fire_cells/fire_cells_{count}.png')

    os.chdir(cwd)


"""
Make a gif from files in a specified folder
"""
def makeGIF(gif_name='fire_cells', gif_dir='../figures/gif_files/fire_cells', fps=1):

    # Build GIF
    print(f'\n\nCreating Gif from: \'{gif_dir}\' ')

    filenames = sorted(glob.glob(os.path.join(gif_dir, '*')))
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(filename))

    imageio.mimwrite(f'../figures/{gif_name}.gif', frames, format='GIF', fps=fps)

    print('\t GIF complete\n')
