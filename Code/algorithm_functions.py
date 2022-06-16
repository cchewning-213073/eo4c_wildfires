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
import skimage.util as ski


from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt


"""
Compute Initial Thresholds:
    MIR, TIR, diff MIR, zenith -  Sentinel Data
"""


def findPotentialFirePixels(data_dir='s3_data', thresholds={'MIR': 310, 'DIF': 10, 'DAY_NIGHT_ZENITH': 85}):

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
        DIF = MIR-TIR
        zenith = np.load(os.path.join(folder, 'solar_zenith_angle.npy'))
        cloud_mask = np.load(os.path.join(folder, 'cloud_mask.npy'))
        water_mask = np.load(os.path.join(folder, 'water_mask.npy'))

        print(zenith.shape)

        # Calculate Thresholds for Fire with MODIS Scheme
        thresh_1 = np.where(MIR > thresholds['MIR'], 1, 0)
        thresh_2 = np.where(DIF > thresholds['DIF'], 1, 0)
        thresh_3 = (thresh_1+thresh_2)*cloud_mask*water_mask

        potential_fire_pixel = np.where((thresh_3) == 2, 1, 0)

        # See if pixel is day or night
        day_pixel = np.where(zenith > thresholds['DAY_NIGHT_ZENITH'], 1, 0)

        # Save file
        print('\tSaving thresholds')
        np.save(os.path.join(folder, 'DIF'), DIF)
        np.save(os.path.join(folder, 'initThresh_MODIS'), potential_fire_pixel)
        np.save(os.path.join(folder, 'day_pixels'), day_pixel)

    os.chdir(cwd)


"""
Create Cloud mask
"""


def maskClouds(data_dir='s3_data'):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Cloud Mask for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'Creating cloud mask for product {count}/{len(path_to_folders)}')

        # Get values
        P_RED_highres = np.load(os.path.join(folder, 'S2_radiance_an.npy'))
        P_NIR_highres = np.load(os.path.join(folder, 'S3_radiance_an.npy'))
        TIR2 = np.load(os.path.join(folder, 'S9_BT_in.npy'))
        day_pixel = np.load(os.path.join(folder, 'day_pixels.npy'))

        # Adjust Radiances to smaller gridsize
        P_RED = ski.view_as_blocks(P_RED_highres, (2, 2)).mean(axis=(2, 3))
        P_NIR = ski.view_as_blocks(P_NIR_highres, (2, 2)).mean(axis=(2, 3))

        # Implement Cloud mask
        # Create grid to hold mask
        y_len = P_RED.shape[0]
        x_len = P_RED.shape[1]

        cloud_mask = np.ones((y_len, x_len))

        for i in range(y_len):
            for j in range(x_len):

                # Mark Nighttime clouds
                if day_pixel[i, j] == 0:
                    if TIR2[i, j] < 265:
                        cloud_mask[i, j] = 0

                # Mark Day time clouds
                else:
                    if P_RED[i, j] + P_NIR[i, j] > 0.9:
                        cloud_mask[i, j] = 0
                    if TIR2[i, j] < 265:
                        cloud_mask[i, j] = 0
                    if (P_RED[i, j] + P_NIR[i, j] > 0.7) and TIR2[i, j] < 285:
                        cloud_mask[i, j] = 0

        print(np.sum(cloud_mask)/cloud_mask.size)

        plt.figure()
        plt.imshow(cloud_mask)
        plt.show()

        np.save(os.path.join(folder, 'cloud_mask'), cloud_mask)

    os.chdir(cwd)


"""
Create Water mask
"""


def maskWater(data_dir='s3_data'):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Water Mask for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'Creating water mask for product {count}/{len(path_to_folders)}')

        # Get values
        P_SWIR_highres = np.load(os.path.join(folder, 'S6_radiance_an.npy'))
        P_NIR_highres = np.load(os.path.join(folder, 'S3_radiance_an.npy'))
        P_RED_highres = np.load(os.path.join(folder, 'S2_radiance_an.npy'))
        # day_pixel = np.load(os.path.join(folder, 'day_pixels.npy'))

        # Adjust Radiances to smaller gridsize
        P_RED = ski.view_as_blocks(P_RED_highres, (2, 2)).mean(axis=(2, 3))
        P_NIR = ski.view_as_blocks(P_NIR_highres, (2, 2)).mean(axis=(2, 3))
        P_SWIR = ski.view_as_blocks(P_SWIR_highres, (2, 2)).mean(axis=(2, 3))

        NDVI = (P_NIR-P_RED)/(P_NIR+P_RED)

        # print(P_RED.min(), P_RED.mean(), P_RED.max())
        # print(P_NIR.min(), P_NIR.mean(), P_NIR.max())
        # print(P_SWIR.min(), P_SWIR.mean(), P_SWIR.max())
        # print(NDVI.min(), NDVI.mean(), NDVI.max())
        #
        # plt.figure()
        # plt.imshow(NDVI)
        # plt.colorbar()
        # plt.title(f'NDVI: {folder}')
        # plt.tight_layout()
        # plt.show()

        # Implement water mask
        # Create grid to hold mask
        y_len = P_RED.shape[0]
        x_len = P_RED.shape[1]

        water_mask = np.ones((y_len, x_len))

        for i in range(y_len):
            for j in range(x_len):
                if (P_SWIR[i, j] < 50) and (P_NIR[i, j] < 150) and (NDVI[i, j] < 0):
                    # if (P_SWIR[i, j] < 0.05) and (P_NIR[i, j] < 0.15) and (NDVI[i, j] < 0):
                    water_mask[i, j] = 0

        print(np.sum(water_mask)/water_mask.size)

        # plt.figure()
        # plt.imshow(water_mask)
        # plt.colorbar()
        # plt.title(f'Water Mask: {folder}')
        # plt.show()

        np.save(os.path.join(folder, 'water_mask'), water_mask)

    os.chdir(cwd)


"""
Create a grid that is specific for the Region of Interest
"""


def createGrid(res=0.05, coordinates={'lon_min': 22, 'lon_max': 24, 'lat_min': 38, 'lat_max': 40}):

    # Define the values that indicate the grid lines
    x_bounds = np.arange(coordinates['lon_min'], (coordinates['lon_max']+(0.5*res)), res)
    y_bounds = np.arange(coordinates['lat_min'], (coordinates['lat_max']+(0.5*res)), res)

    # Define the values that mark the center coordinates of the grid
    x_center = x_bounds[:-1]+(0.5*res)
    y_center = y_bounds[:-1]+(0.5*res)

    # Flip y Values
    y_bounds = np.flip(y_bounds)
    y_center = np.flip(y_center)

    # Create Point Map: then size of the meshed center values, but we label them with an indexe
    # id = np.reshape( np.arange(len(x_center)*len(y_center)), (len(y_center), len(x_center)) )
    num_of_cells = len(y_center)*len(x_center)
    id = np.reshape(np.arange(num_of_cells), (len(y_center), len(x_center)))

    # Create id mapping matrix
    id_index = np.array([(x, y) for y in range(id.shape[0]) for x in range(id.shape[1])])
    index_to_bounds = np.array([(x, y) for y in y_bounds[1:] for x in x_bounds[1:]])
    index_to_center = np.array([(x, y) for y in y_center for x in x_center])

    id_map = np.concatenate((id.reshape(id.size, 1), id_index, index_to_bounds,
                             index_to_center), axis=1)

    # print(id_map[:15, :])

    # plt.figure(figsize=(8, 8))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # plt.scatter(id_map[:, 3], id_map[:, 4], s=20, c=id_map[:, 0], transform=ccrs.PlateCarree())
    # ax.coastlines()
    # ax.gridlines(draw_labels=True)
    # plt.colorbar()
    # plt.show()

    # Wrap it all up for the output
    grid = {
        'lon_min': coordinates['lon_min'],
        'lon_max': coordinates['lon_max'],
        'lat_min': coordinates['lat_min'],
        'lat_max': coordinates['lat_max'],
        'x_bounds': x_bounds,
        'y_bounds': y_bounds,
        'x_center': x_center,
        'y_center': y_center,
        'index_to_bounds': index_to_bounds,
        'id': id,
        'id_map': id_map
    }
    return grid


"""
Assign a Yes or No to each pixel based on if it is in the grid
"""


def assignGridYN(grid_info: dict, data_dir='s3_data'):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')

    # Go into each folder and calculate threshold for each product
    print('\nAssigning a Grid YN to each observation in Product:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'Assigning Grid YN in Product {count}/{len(path_to_folders)}')

        # Get latitude and longitude
        lat = np.load(os.path.join(folder, 'latitude_fn.npy'))
        lon = np.load(os.path.join(folder, 'longitude_fn.npy'))

        # Calculate Conditions
        dummygrid = np.zeros(lat.shape)

        id_count = 0
        for i in range(dummygrid.shape[0]):
            for j in range(dummygrid.shape[1]):

                # Check to see if point is in grid
                if (lat[i, j] < grid_info['lat_min']) or (lat[i, j] > grid_info['lat_max']):
                    dummygrid[i, j] = -1
                elif (lon[i, j] < grid_info['lon_min']) or (lon[i, j] > grid_info['lon_max']):
                    dummygrid[i, j] = -1
                else:
                    dummygrid[i, j] = 1

        # Create array of indices where we have pixels in grid and save
        gridYN = np.where(dummygrid == 1)

        np.save(os.path.join(folder, 'grid_yn'), gridYN)

    os.chdir(cwd)


"""
Assign a Grid ID to each pixel
"""


def assignGridID(grid_info: dict, data_dir='s3_data'):

    id_map = grid_info['id_map']

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')

    # Go into each folder and calculate threshold for each product
    print('\nAssigning a Grid ID to each observation in Product:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'Assigning Grid ID in Product {count}/{len(path_to_folders)}')

        # Get latitude and longitude and gridYN
        gridYN = np.load(os.path.join(folder, 'grid_yn.npy'))
        y = gridYN[0, :]
        x = gridYN[1, :]

        lat = np.load(os.path.join(folder, 'latitude_fn.npy'))
        lon = np.load(os.path.join(folder, 'longitude_fn.npy'))

        lat = lat[y, x]
        lon = lon[y, x]

        # Assign an id to each point
        num_of_pixels = len(lat)
        pixel_cellid = np.zeros(num_of_pixels)
        pixel_assigned = np.zeros(num_of_pixels)
        num_of_cells = id_map.shape[0]

        # Loop through each cell
        for cell in range(num_of_cells):

            # Check each pixel to see if it goes into the cell we are looking at
            for pixel in range(num_of_pixels):

                # Check if we have not assigned the pixel a cell id
                if(pixel_assigned[pixel] == 0):

                    # Check if the pixel is less than the bounds
                    if(lon[pixel] < id_map[cell, 3]):
                        if(lat[pixel] > id_map[cell, 4]):
                            pixel_cellid[pixel] = cell
                            pixel_assigned[pixel] = 1

        pixel_cell_info = np.concatenate(
            (pixel_cellid.reshape(pixel_cellid.size, 1),
             x.reshape(x.size, 1),
             y.reshape(y.size, 1),
             lon.reshape(lon.size, 1),
             lat.reshape(lat.size, 1)), axis=1)
        print('Information on Pixels Cell Assignment:')
        print('Cell ID \t x \t y \t lon \t lat')
        print(pixel_cell_info)

        np.save(os.path.join(folder, 'pixel_cell_info'), pixel_cell_info)

        # Visualize
        # plt.figure(figsize=(10, 10))
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # plt.scatter(lon, lat, s=10, c=pixel_cellid, transform=ccrs.PlateCarree())
        # plt.show()

    os.chdir(cwd)


def makeCellAverage(grid_info: dict, data_dir='s3_data'):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = glob.glob('*')

    # Go into each folder and calculate threshold for each product
    print('\nAggregating Pixel Values Within Cell:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'Aggregating Pixel Values in Product {count}/{len(path_to_folders)}')

        # Get Data From folders
        MIR = np.load(os.path.join(folder, 'F1_BT_fn.npy'))
        TIR = np.load(os.path.join(folder, 'F2_BT_in.npy'))
        DIF = np.load(os.path.join(folder, 'DIF.npy'))
        potential_fire_pixel = np.load(os.path.join(folder, 'initThresh_MODIS.npy'))
        pixel_cell_info = np.load(os.path.join(folder, 'pixel_cell_info.npy'))

        # Aggregate Pixels for Cell value
        num_of_pixels = pixel_cell_info.shape[0]
        num_of_cells = grid_info['id_map'].shape[0]

        print(num_of_pixels)
        print(num_of_cells)

        cell_MIR_avg = np.ones(num_of_cells)*-1000000
        cell_ACTIVE = np.zeros(num_of_cells)

        # Loop through each cell
        for cell in range(num_of_cells):

            # Look at pixel_cell_info
            pixels = (pixel_cell_info[:, 0] == cell)

            y = (pixel_cell_info[pixels, 2].astype(int))
            x = (pixel_cell_info[pixels, 1].astype(int))

            MIR_sum = np.sum(MIR[y, x])
            cell_MIR_avg[cell] = MIR_sum/np.sum(pixels)

            if np.sum(potential_fire_pixel[y, x]) > 0:
                cell_ACTIVE[cell] = 1

        # Save calculations
        # cell_value= np.concatenate((np.arange(num_of_cells).reshape(
        #     num_of_cells, 1), cell_MIR_avg.reshape(cell_MIR_avg.size, 1)), axis=1)
        cell_value = np.concatenate((np.arange(num_of_cells).reshape(
            num_of_cells, 1), cell_ACTIVE.reshape(cell_ACTIVE.size, 1)), axis=1)

        np.save(os.path.join(folder, 'cell_value'), cell_value)

    os.chdir(cwd)
