"""
Functions to load and prepare data for analysis
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
from scipy.interpolate import griddata
import skimage.util as ski

from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt


def lookAtData(data_dir='s3_data', basedir='S3A_SL_1_RBT____20210806T081630_20210806T081930_20210807T160928_0179_075_021_2340_LN2_O_NT_004.SEN3'):

    cwd = os.getcwd()
    os.chdir(data_dir)

    files = glob.glob(basedir+os.sep+'FRP_in.nc')
    for ff in files:
        print(ff)
    print()

    dummyFile = files[0]  # Looking at the first datafile

    with Dataset(dummyFile) as src:

        # Each datafile contains multiple bands
        for band, variable in src.variables.items():
            print('------' + band + '------')

            if "S2_solar_irradiance_an" in band:
                bandName = band
                # Printing the attributes of the BT_in variables:
                for attrname in variable.ncattrs():
                    print("{} -- {}".format(attrname, getattr(variable, attrname)))

    os.chdir(cwd)


"""
Get Sentinel Data
"""


def getSentSatData(sentinelsat_options: dict):
    # Make data directory (if it does not exist).
    os.makedirs(os.path.join(os.getcwd(), sentinelsat_options['data_dir']), exist_ok=True)

    # Initialize API.
    api = SentinelAPI(sentinelsat_options['username'], sentinelsat_options['password'],
                      api_url='https://scihub.copernicus.eu/dhus/')
    print('API Initialized.')

    # Get geojson footprint.
    footprint = geojson_to_wkt(read_geojson(sentinelsat_options['footprint_file']))

    # Find matching files based on criteria.
    products = api.query(area=footprint,
                         date=sentinelsat_options['date'],
                         platformname=sentinelsat_options['platformname'],
                         producttype=sentinelsat_options['producttype'],
                         processinglevel=sentinelsat_options['processinglevel']
                         )

    # convert to Pandas DataFrame
    print('\nRetrieving Products.')
    products_df = api.to_dataframe(products)
    # print(products_df.title)

    if len(products_df) < 1:
        exit('No products available.')

    print(f"\n\nNumber of products available: {len(products_df)}")
    cwd = os.getcwd()  # remember current work directory (CWD).
    os.chdir(sentinelsat_options['data_dir'])  # Change directory.
    #
    # Select and download product to data directory.
    # Get desired product based on selected index.
    product_df = products_df.head(sentinelsat_options['download_index'])
    # Download product. Needs '.index' as it cannot download df directly.
    api.download_all(product_df.index)
    # print(f"\nDownloading product: {products_df.head(index)['title']}")

    # Unzip files
    path_to_zip_file = sorted(glob.glob('*.zip'))
    for count in range(sentinelsat_options['download_index']):

        with zipfile.ZipFile(path_to_zip_file[count], 'r') as zip_ref:
            zip_ref.extractall()

        # Delete Zip file
        os.remove(path_to_zip_file[count])

    os.chdir(cwd)  # Return to project directory.


def getSent2SatData(sentinelsat_options: dict):
    # Make data directory (if it does not exist).
    os.makedirs(os.path.join(os.getcwd(), sentinelsat_options['data_dir']), exist_ok=True)

    # Initialize API.
    api = SentinelAPI(sentinelsat_options['username'], sentinelsat_options['password'],
                      api_url='https://scihub.copernicus.eu/dhus/')
    print('API Initialized.')

    # Get geojson footprint.
    footprint = geojson_to_wkt(read_geojson(sentinelsat_options['footprint_file']))

    # Find matching files based on criteria.
    products = api.query(area=footprint,
                         date=sentinelsat_options['date'],
                         platformname=sentinelsat_options['platformname'],
                         producttype=sentinelsat_options['producttype'],
                         # processinglevel=sentinelsat_options['processinglevel']
                         )

    # convert to Pandas DataFrame
    print('\nRetrieving Products.')
    products_df = api.to_dataframe(products)
    # print(products_df.title)

    if len(products_df) < 1:
        exit('No products available.')

    print(f"\n\nNumber of products available: {len(products_df)}")
    cwd = os.getcwd()  # remember current work directory (CWD).
    os.chdir(sentinelsat_options['data_dir'])  # Change directory.
    #
    # Select and download product to data directory.
    # Get desired product based on selected index.
    product_df = products_df.head(sentinelsat_options['download_index'][0])
    # Download product. Needs '.index' as it cannot download df directly.
    api.download_all(product_df.index)
    # print(f"\nDownloading product: {products_df.head(index)['title']}")

    # Unzip files
    path_to_zip_file = sorted(glob.glob('*.zip'))
    for count in range(sentinelsat_options['download_index']):

        with zipfile.ZipFile(path_to_zip_file[count], 'r') as zip_ref:
            zip_ref.extractall()

        # Delete Zip file
        os.remove(path_to_zip_file[count])

    os.chdir(cwd)  # Return to project directory.


"""
Return Data necessary for Initial Thresholds:
    MIR, TIR, diff MIR, zenith -  Sentinel Data
"""


def getProducts(data_dir='s3_data', file_list = []):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(data_dir)

    # Get list of available folders
    path_to_folders = sorted(glob.glob('*.SEN3'))
    print('\nAvailable Folders:')
    for folder in path_to_folders:
        print(folder)

    # Go into each folder and put needed bands into new folder for later reference
    print('\nExtracting Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tExtracting Product {count}/{len(path_to_folders)}')

        # Get product date - S3A_SL_1_RBT____20210809T195551_20210809T195851_20210811T075833_0180_075_071_0540_LN2_O_NT_004.SEN3
        product_sat = folder[:3]
        product_year = folder[16:20]
        product_month = folder[20:22]
        product_day = folder[22:24]
        product_time = folder[25:31]
        new_folder_name = product_day+'_'+product_month+'_'+product_year+'_T_'+product_time+'_'+product_sat

        # Create folder in inputs to hold the created files
        os.makedirs(os.path.join('inputs', new_folder_name), exist_ok=True)


        for file in file_list:

            print(f'\t\tMoving file {file}')
            file_path = glob.glob(folder+os.sep+file)[0]

            with Dataset(file_path) as src:
                # Each datafile contains multiple bands
                for band, variable in src.variables.items():

                    # print(band, variable)

                    # Get Brightness Temps
                    if file[:8] in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        bandData = data.variables[bandName][:]
                        data.close()
                        # Save file
                        np.save(os.path.join(os.path.join('inputs', new_folder_name), bandName), bandData)

                    # Get Radiances
                    if file[:14] in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        bandData = data.variables[bandName][:]
                        data.close()
                        # Save file
                        np.save(os.path.join(os.path.join('inputs', new_folder_name), bandName), bandData)

                    # Get coordinates
                    if file[:11] == 'geodetic_fn':
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        lat_fn = data.variables['latitude_fn'][:]
                        lon_fn = data.variables['longitude_fn'][:]
                        data.close()
                        # Save file
                        np.save(os.path.join(os.path.join(
                            'inputs', new_folder_name), 'latitude_fn'), lat_fn)
                        np.save(os.path.join(os.path.join(
                            'inputs', new_folder_name), 'longitude_fn'), lon_fn)

                    # Get Zenith and Azmith values
                    if 'solar_zenith_tn' in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        solar_zenith_tn = data.variables[bandName][:]
                        data.close()

                    if 'sat_zenith_tn' in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        sat_zenith_tn = data.variables[bandName][:]
                        data.close()

                    if 'solar_azimuth_tn' in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        solar_azimuth_tn = data.variables[bandName][:]
                        data.close()

                    if 'sat_azimuth_tn' in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        sat_azimuth_tn = data.variables[bandName][:]
                        data.close()

                    if file[:11] == 'geodetic_tx':
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        lat_tx = data.variables['latitude_tx'][:]
                        lon_tx = data.variables['longitude_tx'][:]
                        data.close()

                    if 'solar_irradiance_an' in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        solar_irradiance_an = data.variables[bandName][:]
                        data.close()
                        np.save(os.path.join(os.path.join('inputs', new_folder_name), bandName), solar_irradiance_an)

                    if 'flags' in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        FRP = data.variables[bandName][:]
                        data.close()
                        np.save(os.path.join(os.path.join('inputs', new_folder_name), bandName), FRP)

        # Take zenith inputs and expand grid to what we want
        if (data_dir=='s3_data'):
            solar_zenith_angle = griddata((lat_tx.flatten(), 
                                           lon_tx.flatten()),
                                           solar_zenith_tn.flatten(), 
                                           (lat_fn, lon_fn), 
                                           method='linear')

            sat_zenith_angle = griddata((lat_tx.flatten(), 
                                         lon_tx.flatten()),
                                         sat_zenith_tn.flatten(), 
                                         (lat_fn, lon_fn), 
                                         method='linear')

            solar_azimuth_angle = griddata((lat_tx.flatten(), 
                                         lon_tx.flatten()),
                                         solar_azimuth_tn.flatten(), 
                                         (lat_fn, lon_fn), 
                                         method='linear')

            sat_azimuth_angle = griddata((lat_tx.flatten(), 
                                         lon_tx.flatten()),
                                         sat_azimuth_tn.flatten(), 
                                         (lat_fn, lon_fn), 
                                         method='linear')

            # Calculate Glint
            glint_angle = np.cos(sat_zenith_angle)*np.cos(solar_zenith_angle) - np.sin(sat_zenith_angle)*np.sin(solar_zenith_angle)*np.cos(np.abs(solar_azimuth_angle-sat_azimuth_angle))


            np.save(os.path.join(os.path.join('inputs', new_folder_name), 'solar_zenith_angle'), solar_zenith_angle)
            np.save(os.path.join(os.path.join('inputs', new_folder_name), 'glint_angle'), glint_angle)
            # np.save(os.path.join(os.path.join('inputs', new_folder_name), 'sat_zenith_angle'), sat_zenith_angle)
            # np.save(os.path.join(os.path.join('inputs', new_folder_name), 'solar_azimuth_angle'), solar_azimuth_angle)
            # np.save(os.path.join(os.path.join('inputs', new_folder_name), 'sat_azimuth_angle'), sat_azimuth_angle)

    # Once done with everything, return to cwd
    os.chdir(cwd)


"""
Calculate Reflectance From radiance:
"""


def calcReflectance(data_dir='s3_data'):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(os.path.join(data_dir, 'inputs'))

    # Get list of available folders
    path_to_folders = sorted(glob.glob('*'))

    # Go into each folder and calculate threshold for each product
    print('\nCalculating Reflectance for Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'\n\tCalculating reflectance for product {count}/{len(path_to_folders)}')

        # inport files
        L_RED = np.load(os.path.join(folder, 'S2_radiance_an.npy'))
        L_NIR = np.load(os.path.join(folder, 'S3_radiance_an.npy'))
        L_SWIR = np.load(os.path.join(folder, 'S6_radiance_an.npy'))

        L_RED = ski.view_as_blocks(L_RED, (2, 2)).mean(axis=(2, 3))
        L_NIR = ski.view_as_blocks(L_NIR, (2, 2)).mean(axis=(2, 3))
        L_SWIR = ski.view_as_blocks(L_SWIR, (2, 2)).mean(axis=(2, 3))

        I_RED = np.load(os.path.join(folder, 'S2_solar_irradiance_an.npy'))[0]
        I_NIR = np.load(os.path.join(folder, 'S3_solar_irradiance_an.npy'))[0]
        I_SWIR = np.load(os.path.join(folder, 'S6_solar_irradiance_an.npy'))[0]

        solar_zenith = np.load(os.path.join(folder, 'solar_zenith_angle.npy'))

        # Calculate Reflectance
        P_RED = np.pi * (L_RED / I_RED / np.cos(solar_zenith))
        P_NIR = np.pi * (L_NIR / I_NIR / np.cos(solar_zenith))
        P_SWIR = np.pi * (L_SWIR / I_SWIR / np.cos(solar_zenith))

        # Save
        np.save(f'{folder}/S2_reflectance_an', P_RED)
        np.save(f'{folder}/S3_reflectance_an', P_NIR)
        np.save(f'{folder}/S6_reflectance_an', P_SWIR)

    os.chdir(cwd)
