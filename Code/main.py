"""
    30574 Earth observations for monitoring changes (EO4Change) Wildfire and Biomass Impacts.
"""
##### Imports
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
import glob

# User defined functions
from algorithm_functions import *
from display_functions import *
from prep_functions import *
from grid_functions import *


print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Running Main File:')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')



##### Define Data product Dictionaries

SENTINELSAT_DATE = ['20210806', '20210810']

SENTINELSAT_OPTIONS = {
    # Start, End date, YYYYMMDD, must be string format. The summer period is adviced.
    'date': SENTINELSAT_DATE,
    'platformname': 'Sentinel-3',  # Satellite platform name.
    'producttype': 'SL_1_RBT___',  # Find products @ link below:
    'processinglevel': '1',  # Level of the product.
    'processingmode': 'Offline',  # Processing mode, i.e. Offline, NRT.
    'footprint_file': 'greece.geojson',
    'username': 'connorchewning',
    'password': 'Denmark21',
    'data_dir': 's3_data',  # Directory to download data to.
    # Must be > 1. Multiple indexes list will download multiple available files.
    'download_index': [12],
}

SENTINELSAT_DATE = ['20210803', '20210817']

SENTINELSAT2_OPTIONS = {
    # Start, End date, YYYYMMDD, must be string format. The summer period is adviced.
    'date': SENTINELSAT_DATE,
    'platformname': 'Sentinel-2',  # Satellite platform name.
    'producttype': 'S2MSI2A',  # Find products @ link below:
    'processinglevel': '2A',  # Level of the product.
    'processingmode': 'Offline',  # Processing mode, i.e. Offline, NRT.
    'footprint_file': 'greece.geojson',
    'username': 'connorchewning',
    'password': 'Denmark21',
    'data_dir': 's2_data',  # Directory to download data to.
    # Must be > 1. Multiple indexes list will download multiple available files.
    'download_index': 12,
}


# Get S3 Data
# getSentSatData(sentinelsat_options=SENTINELSAT_OPTIONS)

# Get S2 Data
# getSentSatData(sentinelsat_options=SENTINELSAT2_OPTIONS)

# Used to look at files in S3 folders
lookAtData()

# Display data
# displayData(sentinelsat_options=SENTINELSAT_OPTIONS)
# makeGIF('f1_band', '../figures/gif_files/f1_band', 0.5)


# Get Needed Band Data in usable format
# getProducts(data_dir=SENTINELSAT_OPTIONS['data_dir'])

# Calculate the reflectance from the radiance values
# calcReflectance(data_dir=SENTINELSAT_OPTIONS['data_dir'])


# Compute Initial MODIS Thresholds
INITIAL_THRESHOLDS = {
    'MIR': 310,
    'DIF': 10,
    'DAY_NIGHT_ZENITH': 85
}
# markDayPixels(data_dir='s3_data', thresholds={'DAY_NIGHT_ZENITH': 85})
#
# # Mask Clouds
# maskClouds(data_dir=SENTINELSAT_OPTIONS['data_dir'])
# makeGIF('cloud_mask', '../figures/gif_files/cloud_mask', 0.5)
#
# # Mask Water
# maskWater(data_dir=SENTINELSAT_OPTIONS['data_dir'])
# # makeGIF('water_mask', '../figures/gif_files/water_mask', 0.5)
# makeGIF('NDVI', '../figures/gif_files/NDVI', 0.5)

# Find the pixels that may hold active fires
# findPotentialFirePixels(data_dir=SENTINELSAT_OPTIONS['data_dir'], thresholds=INITIAL_THRESHOLDS)
# makeGIF('potential_fires_FULL', '../figures/gif_files/potential_fires_FULL', 0.5)

# Display Potential Fire displayFirePixels
COORDINATES = {
    'lon_min': 22,
    'lon_max': 24,
    'lat_min': 38,
    'lat_max': 40
}
# displayFirePixels(data_dir=SENTINELSAT_OPTIONS['data_dir'], coordinates=COORDINATES, show=False, confirmed=False)
# makeGIF('potential_fire_pixels', '../figures/gif_files/potential_fire_pixels', 0.5)


# calculateBackground(data_dir='s3_data')

# confirmFirePixels()
# makeGIF('confirmed_fire_pixels_full', '../figures/gif_files/confirmed_fire_pixels_full', 0.5)

# displayFirePixels(data_dir=SENTINELSAT_OPTIONS['data_dir'], coordinates=COORDINATES, show=False, confirmed=True)
# makeGIF('confirmed_fire_pixels', '../figures/gif_files/confirmed_fire_pixels', 0.5)

# # Create Grid
# RESOLUTION = 0.05
# grid_info = createGrid(res=RESOLUTION, coordinates=COORDINATES)

#  # assign points to grid
# assignGridYN(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])
# makeGIF('grid_location', '../figures/gif_files/grid_location', 0.5)

# # Decrease Resolution based on gridYN
# assignGridID(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])
# makeGIF('grid', '../figures/gif_files/grid', 0.5)

# # Make cell averages
# makeCellAverage(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])

# # Show cell averages
# displayCellValue(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])
# makeGIF('fire_cells', '../figures/gif_files/fire_cells', 0.5)
