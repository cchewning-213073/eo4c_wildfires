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


def markDayPixels(data_dir='s3_data', thresholds={'DAY_NIGHT_ZENITH': 85}):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and calculate threshold for each product
    print('\nMarking Day Pixels for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tMarking Day Pixels for product {count}/{len(path_to_folders)}')

        zenith = np.load(os.path.join(folder, 'solar_zenith_angle.npy'))

        # See if pixel is day or night
        day_pixel = np.where(zenith > thresholds['DAY_NIGHT_ZENITH'], 1, 0)

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
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Cloud Mask for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tCreating cloud mask for product {count}/{len(path_to_folders)}')

        # Get values
        P_RED = np.load(os.path.join(folder, 'S2_reflectance_an.npy'))
        P_NIR = np.load(os.path.join(folder, 'S3_reflectance_an.npy'))
        TIR2 = np.load(os.path.join(folder, 'S9_BT_in.npy'))
        day_pixel = np.load(os.path.join(folder, 'day_pixels.npy'))

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
        # plt.show()
        plt.savefig(f'../../../figures/gif_files/cloud_mask/cloud_mask_{count}.png')
        plt.close()

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
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Water Mask for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tCreating water mask for product {count}/{len(path_to_folders)}')

        # Get values
        P_RED = np.load(os.path.join(folder, 'S2_reflectance_an.npy'))
        P_NIR = np.load(os.path.join(folder, 'S3_reflectance_an.npy'))
        P_SWIR = np.load(os.path.join(folder, 'S6_reflectance_an.npy'))

        NDVI = (P_NIR-P_RED)/(P_NIR+P_RED)

        plt.figure()
        plt.imshow(NDVI, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f'NDVI: {folder}')
        plt.tight_layout()
        plt.savefig(f'../../../figures/gif_files/NDVI/NDVI_{count}.png')
        plt.close()
        # plt.show()

        # Implement water mask
        # Create grid to hold mask
        y_len = P_RED.shape[0]
        x_len = P_RED.shape[1]

        water_mask = np.ones((y_len, x_len))

        for i in range(y_len):
            for j in range(x_len):
                ##### NEED TO FIX RADIANCE TO REFLECTANCE #####
                if (P_SWIR[i, j] < .05) and (P_NIR[i, j] < .15) and (NDVI[i, j] < 0):
                    # if (P_SWIR[i, j] < 0.05) and (P_NIR[i, j] < 0.15) and (NDVI[i, j] < 0):
                    water_mask[i, j] = 0

        print(np.sum(water_mask)/water_mask.size)

        plt.figure()
        plt.imshow(water_mask)
        plt.colorbar()
        plt.title(f'Water Mask: {folder}')
        plt.savefig(f'../../../figures/gif_files/water_mask/water_mask_{count}.png')
        plt.close()

        np.save(os.path.join(folder, 'water_mask'), water_mask)

    os.chdir(cwd)


"""
findPotentialFirePixels:
    MIR, TIR, diff MIR, zenith -  Sentinel Data
"""
def findPotentialFirePixels(data_dir='s3_data', thresholds={'MIR': 310, 'DIF': 10, 'DAY_NIGHT_ZENITH': 85}):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Potential Fire Pixels for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tCalculating Potential Fire Pixels for product {count}/{len(path_to_folders)}')

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

        potential_fire_pixels = np.where((thresh_3) == 2, 1, 0)

        # Save file
        print('\tSaving thresholds')
        np.save(os.path.join(folder, 'DIF'), DIF)
        np.save(os.path.join(folder, 'potential_fire_pixels'), potential_fire_pixels)

        plt.figure()
        plt.imshow(potential_fire_pixels)
        plt.colorbar()
        plt.title(f'potential_fires_FULL: {folder}')
        plt.savefig(f'../../../figures/gif_files/potential_fires_FULL/potential_fires_FULL_{count}.png')
        plt.close()

    os.chdir(cwd)


"""
Calculate a potential fire pixels background value
"""
def calculateBackground(data_dir='s3_data'):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Potential Fire Pixel Background Value for Products:')
    path_to_folders = sorted(glob.glob('*'))
    count = 0
    for folder in path_to_folders:
        count += 1
        print(
            f'\n\tCalculating Potential Fire Pixel Background Value  for product {count}/{len(path_to_folders)}')

        # Bring in potential Fire Pixel
        potential_fire_pixels = np.load(os.path.join(folder, 'potential_fire_pixels.npy'))

        MIR = np.load(os.path.join(folder, 'F1_BT_fn.npy'))
        TIR = np.load(os.path.join(folder, 'F2_BT_in.npy'))
        DIF = np.load(os.path.join(folder, 'DIF.npy'))
        RED = np.load(os.path.join(folder, 'S2_reflectance_an.npy'))

        # Bring in masks and sun glint info
        cloud_mask = np.load(os.path.join(folder, 'cloud_mask.npy'))
        water_mask = np.load(os.path.join(folder, 'water_mask.npy'))
        glint_angle = np.load(os.path.join(folder, 'glint_angle.npy'))

        # Make valid_pixels (1= Valid, 0= not valid)
        # PAGE 32 ADD OTHER VALID CONDITIONS!!!!!
        total_mask = cloud_mask*water_mask

        # Make empty arrays to hold background mean and std
        pixel_background_mean_MIR = np.zeros(potential_fire_pixels.shape)
        pixel_background_mean_DIF = np.zeros(potential_fire_pixels.shape)
        pixel_background_mean_TIR = np.zeros(potential_fire_pixels.shape)
        pixel_background_mean_RED = np.zeros(potential_fire_pixels.shape)

        pixel_background_std_MIR  = np.zeros(potential_fire_pixels.shape)
        pixel_background_std_DIF  = np.zeros(potential_fire_pixels.shape)
        pixel_background_std_TIR  = np.zeros(potential_fire_pixels.shape)

        pixel_backgroundfire_mean_MIR = np.zeros(potential_fire_pixels.shape)
        pixel_backgroundfire_std_MIR  = np.zeros(potential_fire_pixels.shape)


        # Define grid width sizes
        grid_widths = np.arange(5, 22, 2)

        # Loop through potential fire pixels
        y_len = cloud_mask.shape[0]
        x_len = cloud_mask.shape[1]

        for i in range(y_len):
            for j in range(x_len):

                if(i in np.arange(0, 1500, 50).astype(int) and j == 0):
                    print(f'\t\tPixel Row:{i}')

                # If a fire has been marked
                if(potential_fire_pixels[i, j] == 1):

                    # Establish counts
                    num_background_pixels_inspected = 0    # num of pixels we have looked at for this fire pixel
                    num_background_pixels_valid = 0        # number of valid Pixel
                    percent_valid = 0
                    valid_background_MIR_values = np.array([])
                    valid_background_DIF_values = np.array([])
                    valid_background_TIR_values = np.array([])
                    valid_background_RED_values = np.array([])

                    backgroundfire_MIR_values   = np.array([])


                    # Loop through grids
                    grid_count = 0  # represnted the grid size that we want to look at
                    while (percent_valid < 0.25) and (grid_count < len(grid_widths)):

                        if (grid_count < len(grid_widths)):

                            # Set the current grid width and delta "bounds"
                            current_grid_width = grid_widths[grid_count]
                            delta = (current_grid_width-1)/2

                            # print(f'\t\tCurrent Grid Size: {current_grid_width}x{current_grid_width}')

                            # Create "Mini Grid" -> outlines the background area
                            mini_y = np.arange(i-delta, i+delta).astype(int)
                            mini_x = np.arange(j-delta, j+delta).astype(int)
                            # mini_grid = np.array([(i, j) for i in mini_y for j in mini_x])
                            for ii in mini_y:
                                for jj in mini_x:
                                    # Check if backgroud_pixel falls within actual image bounds
                                    if (ii >= 0) and (jj >= 0) and (ii < y_len) and (jj < x_len):
                                        # Check if the pixel is right next to fire pixels
                                        if ((ii == i-1) or (ii == i) or (ii == i+1)) and ((jj == j-1) or (jj == j) or (jj == j+1)):
                                            num_background_pixels_inspected += 1
                                            # Check if the pixel is a water or cloud pixel
                                            if (total_mask[ii, jj] == 1):
                                                    #### Now perform checks from the paper
                                                    # Check if this pixel is not a "Background fire pixel" Check 12c and 12d
                                                    if (MIR[ii,jj] < 325) and (DIF[ii,jj] < 20):
                                                        #  12a)                        12b)                    12e) SUNGLINT CONDITION
                                                        if (MIR[ii,jj] < MIR[i,j]) and (DIF[ii,jj] < DIF[i,j]) and (glint_angle[i,j] < 2):

                                                            # This is a good background pixel!
                                                            num_background_pixels_valid += 1

                                                            # add the value of this pixel to the value arrays
                                                            valid_background_MIR_values = np.append(valid_background_MIR_values, MIR[ii, jj])
                                                            valid_background_TIR_values = np.append(valid_background_TIR_values, TIR[ii, jj])
                                                            valid_background_DIF_values = np.append(valid_background_DIF_values, DIF[ii, jj])
                                                            valid_background_RED_values = np.append(valid_background_RED_values, RED[ii, jj])

                                                    # If it is a background fire pixel, we save these values for calculations later
                                                    else:
                                                        backgroundfire_MIR_values = np.append(backgroundfire_MIR_values, MIR[ii, jj])

                            # Once we have looked at all of the pixels, calculate value statistics of background for fire pixel
                            if num_background_pixels_valid > 0:
                                # print('\t\tAssigning Background to fire pixel')
                                pixel_background_mean_MIR[i, j] = np.mean(valid_background_MIR_values)
                                pixel_background_mean_TIR[i, j] = np.mean(valid_background_TIR_values)
                                pixel_background_mean_DIF[i, j] = np.mean(valid_background_DIF_values)
                                pixel_background_mean_RED[i, j] = np.mean(valid_background_RED_values)

                                pixel_background_std_MIR[i, j] = np.std(valid_background_MIR_values)
                                pixel_background_std_DIF[i, j] = np.std(valid_background_DIF_values)
                                pixel_background_std_TIR[i, j] = np.std(valid_background_TIR_values)

                                percent_valid = num_background_pixels_valid / num_background_pixels_inspected

                            if len(backgroundfire_MIR_values) > 0:
                                pixel_backgroundfire_mean_MIR[i, j] = np.mean(backgroundfire_MIR_values)
                                pixel_backgroundfire_std_MIR[i, j]  = np.std(backgroundfire_MIR_values)

                        else:
                            # print('\t\tNo Valid Grid Cell Size for background calculation')
                            pixel_background_mean_MIR[i, j] = -100000
                            pixel_background_mean_DIF[i, j] = -100000
                            pixel_background_mean_TIR[i, j] = -100000
                            pixel_background_mean_RED[i, j] = -100000

                            pixel_background_std_MIR[i, j]  = -100000
                            pixel_background_std_DIF[i, j]  = -100000
                            pixel_background_std_TIR[i, j]  = -100000

                        # Increase count
                        grid_count += 1

        np.save(os.path.join(folder, 'pixel_background_mean_MIR'), pixel_background_mean_MIR)
        np.save(os.path.join(folder, 'pixel_background_mean_TIR'), pixel_background_mean_TIR)
        np.save(os.path.join(folder, 'pixel_background_mean_DIF'), pixel_background_mean_DIF)
        np.save(os.path.join(folder, 'pixel_background_mean_RED'), pixel_background_mean_RED)

        np.save(os.path.join(folder, 'pixel_background_std_MIR'), pixel_background_std_MIR)
        np.save(os.path.join(folder, 'pixel_background_std_DIF'), pixel_background_std_DIF)
        np.save(os.path.join(folder, 'pixel_background_std_TIR'), pixel_background_std_TIR)

        np.save(os.path.join(folder, 'pixel_backgroundfire_mean_MIR'), pixel_backgroundfire_mean_MIR)
        np.save(os.path.join(folder, 'pixel_backgroundfire_std_MIR'), pixel_backgroundfire_std_MIR)

        # plt.figure()
        # plt.imshow(pixel_background_value_MIR)
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(pixel_background_value_DIF)
        # plt.colorbar()
        # plt.show()

    os.chdir(cwd)


"""
Confirm the previously flagged pixels with the background values
"""
def confirmFirePixels(data_dir='s3_data', thresholds={}):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Go into each folder and calculate threshold for each product
    print('\nConfirming Fire Pixels for Products:')
    path_to_folders = sorted(glob.glob('*'))
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tConfirming Fire Pixels for product {count}/{len(path_to_folders)}')

        # Bring in potential Fire Pixels and the background values
        potential_fire_pixels = np.load(os.path.join(folder, 'potential_fire_pixels.npy'))
        MIR = np.load(os.path.join(folder, 'F1_BT_fn.npy'))
        TIR = np.load(os.path.join(folder, 'F2_BT_in.npy'))
        DIF = np.load(os.path.join(folder, 'DIF.npy'))
        RED = np.load(os.path.join(folder, 'S2_reflectance_an.npy'))
        glint_angle = np.load(os.path.join(folder, 'glint_angle.npy')) 

        pixel_background_mean_MIR = np.load(os.path.join(folder, 'pixel_background_mean_MIR.npy'))
        pixel_background_mean_TIR = np.load(os.path.join(folder, 'pixel_background_mean_TIR.npy'))
        pixel_background_mean_DIF = np.load(os.path.join(folder, 'pixel_background_mean_DIF.npy'))
        pixel_background_mean_RED = np.load(os.path.join(folder, 'pixel_background_mean_RED.npy'))

        pixel_background_std_MIR = np.load(os.path.join(folder, 'pixel_background_std_MIR.npy'))
        pixel_background_std_DIF = np.load(os.path.join(folder, 'pixel_background_std_DIF.npy'))
        pixel_background_std_TIR = np.load(os.path.join(folder, 'pixel_background_std_TIR.npy'))

        pixel_backgroundfire_mean_MIR = np.load(os.path.join(folder, 'pixel_backgroundfire_mean_MIR.npy'))
        pixel_backgroundfire_std_MIR  = np.load(os.path.join(folder, 'pixel_backgroundfire_std_MIR.npy'))

        # Make holder for confirmed Pixel
        confirmed_fire_pixels = np.zeros(potential_fire_pixels.shape)

        # Thresholds
        y_len = potential_fire_pixels.shape[0]
        x_len = potential_fire_pixels.shape[1]
        for i in range(y_len):
            for j in range(x_len):

                # If a fire has been marked
                if(potential_fire_pixels[i, j] == 1):

                    ##### Absolute Thresholds #####
                    if(MIR[i, j] > 360):
                        confirmed_fire_pixels[i, j] = 1

                    ##### Context Thresholds #####
                    # Context Thres 14a
                    if ((RED[i, j] / MIR[i, j]) > (pixel_background_mean_RED[i, j] / pixel_background_mean_MIR[i, j])):
                        # 14b)
                        if (DIF[i,j] > pixel_background_mean_DIF[i,j] + 3.2*pixel_background_std_DIF[i,j] ):
                            # 14c) - Maggie's Favorite
                            if (DIF[i, j] > pixel_background_mean_DIF[i, j] + 5.6):
                                # 14d)
                                if (MIR[i,j] > pixel_background_mean_MIR[i,j] + 3*pixel_background_std_MIR[i,j]):
                                    # 14e)
                                    if (TIR[i, j] > pixel_background_mean_TIR[i, j] - 4):
                                        confirmed_fire_pixels[i, j] = 1
                                    # 14f)
                                    elif (pixel_backgroundfire_std_MIR[i,j] > 5):
                                        confirmed_fire_pixels[i, j] = 1


        # Save confirmed fire pixels
        np.save(os.path.join(folder, 'confirmed_fire_pixels'), confirmed_fire_pixels)

        plt.figure()
        plt.imshow(confirmed_fire_pixels)
        plt.colorbar()
        plt.savefig(f'../../../figures/gif_files/confirmed_fire_pixels_full/confirmed_fire_pixels_full_{count}.png')
        plt.close()

    os.chdir(cwd)



"""
Remove False Alarms
"""
def detectFalseAlarms(data_dir='s3_data', thresholds={}):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Go into each folder and calculate threshold for each product
    print('\nDetecting False Alarms for Products:')
    path_to_folders = sorted(glob.glob('*'))
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tDetecting False Alarms for product {count}/{len(path_to_folders)}')

        # Bring in confirmed fire
        confirmed_fire_pixels_FA_removed = np.load(os.path.join(folder, 'confirmed_fire_pixels.npy'))
        # MIR = np.load(os.path.join(folder, 'F1_BT_fn.npy'))
        # TIR = np.load(os.path.join(folder, 'F2_BT_in.npy'))
        # DIF = np.load(os.path.join(folder, 'DIF.npy'))
        RED = np.load(os.path.join(folder, 'S2_reflectance_an.npy'))
        glint_angle = np.load(os.path.join(folder, 'glint_angle.npy')) 


        # Thresholds
        y_len = confirmed_fire_pixels_FA_removed.shape[0]
        x_len = confirmed_fire_pixels_FA_removed.shape[1]
        for i in range(y_len):
            for j in range(x_len):

                # If a fire has been marked
                if(confirmed_fire_pixels_FA_removed[i, j] == 1):

                    ##### False Alarm Elimination #####
                    # print(RED.shape)
                    if (RED[i,j] > 0.15): #or (glint_angle[i,j] < 2) : #### ADD FINAL CONDITION
                        confirmed_fire_pixels_FA_removed[i, j] = 0

        # Save confirmed fire pixels
        np.save(os.path.join(folder, 'confirmed_fire_pixels_FA_removed'), confirmed_fire_pixels_FA_removed)

        plt.figure()
        plt.imshow(confirmed_fire_pixels_FA_removed)
        plt.colorbar()
        plt.savefig(f'../../../figures/gif_files/confirmed_fire_pixels_FA_removed/confirmed_fire_pixels_FA_removed_{count}.png')
        plt.close()

    os.chdir(cwd)