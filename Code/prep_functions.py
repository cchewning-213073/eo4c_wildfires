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

from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt


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
    print(products_df.title)

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
    path_to_zip_file = glob.glob('*.zip')
    for count in range(sentinelsat_options['download_index']):

        with zipfile.ZipFile(path_to_zip_file[count], 'r') as zip_ref:
            zip_ref.extractall()

        # # rename folder
        # zipped_name = path_to_zip_file[count][:-4]
        # path_to_SEN3_folder = (zipped_name+'.SEN3')
        # new_SEN3_name = sentinelsat_options['return_file_prefix'] + \
        #     sentinelsat_options['return_file_prefix'][0]+'_to_' + \
        #     sentinelsat_options['return_file_prefix'][1]+'_Count_'+str(count)+'.SEN3'
        #
        # os.rename(path_to_SEN3_folder, new_SEN3_name)

        # Delete Zip file
        os.remove(path_to_zip_file[count])

    os.chdir(cwd)  # Return to project directory.


"""
Return Data necessary for Initial Thresholds:
    MIR, TIR, diff MIR, zenith -  Sentinel Data
"""


def getThresholdProducts(data_dir='s3_data'):

    # Get current directory and move into the data folder
    cwd = os.getcwd()
    os.chdir(data_dir)

    # Get list of available folders
    path_to_folders = glob.glob('*.SEN3')
    print('Available Folders:')
    for folder in path_to_folders:
        print(folder)

    # Go into each folder and put needed bands into new folder for later reference
    print('\nExtracting Products:')
    count = 0
    for folder in path_to_folders:
        count += 1
        print(f'Extracting Product {count}/{len(path_to_folders)}')

        # Get product date - S3A_SL_1_RBT____20210809T195551_20210809T195851_20210811T075833_0180_075_071_0540_LN2_O_NT_004.SEN3
        product_sat = folder[:2]
        product_year = folder[16:20]
        product_month = folder[20:22]
        product_day = folder[22:24]
        product_time = folder[25:31]
        new_folder_name = product_day+'_'+product_month+'_'+product_year+'_T_'+product_time+'_'+product_sat

        # Create folder in inputs to hold the created files
        os.makedirs(os.path.join('inputs', new_folder_name), exist_ok=True)

        # List of wanted files
        file_list = ['F1_BT_fn.nc', 'F1_BT_fo.nc',
                     'F2_BT_in.nc', 'F2_BT_io.nc',
                     'S7_BT_in.nc', 'S7_BT_io.nc',
                     'S8_BT_in.nc', 'S8_BT_io.nc',
                     'geodetic_fn.nc', 'geodetic_fo.nc',
                     'geodetic_in.nc', 'geodetic_io.nc']

        for file in file_list:

            print(f'\tMoving file {file}')
            file_path = glob.glob(folder+os.sep+file)[0]

            with Dataset(file_path) as src:
                # Each datafile contains multiple bands
                for band, variable in src.variables.items():

                    # Get Brightness Temps
                    if file[:8] in band:
                        bandName = band
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        bandData = data.variables[bandName][:]
                        data.close()
                        # Save file
                        np.save(os.path.join(os.path.join('inputs', new_folder_name), bandName), bandData)

                    # Get coordinates

                    elif file[:11] == 'geodetic_fn':
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        lat = data.variables['latitude_fn'][:]
                        lon = data.variables['longitude_fn'][:]
                        data.close()
                        # Save file
                        np.save(os.path.join(os.path.join('inputs', new_folder_name), 'latitude_fn'), lat)
                        np.save(os.path.join(os.path.join(
                            'inputs', new_folder_name), 'longitude_fn'), lon)

                    # Get coordinates
                    elif file[:9] == 'geodetic_i':
                        data = Dataset(file_path)
                        data.set_auto_mask(False)
                        lat = data.variables['latitude_in'][:]
                        lon = data.variables['longitude_in'][:]
                        data.close()
                        # Save file
                        np.save(os.path.join(os.path.join('inputs', new_folder_name), 'latitude_in'), lat)
                        np.save(os.path.join(os.path.join(
                            'inputs', new_folder_name), 'longitude_in'), lon)

    # Once done with everything, return to cwd
    os.chdir(cwd)
