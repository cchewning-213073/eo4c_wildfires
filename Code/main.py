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
from validation_functions import *

#####################################################################################
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Running Main File:')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
#####################################################################################



##### Define Data product Dictionaries

SENTINELSAT_DATE = ['20210806', '20210810']
DOWNLOAD_INDEX = 12

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
    'download_index': [DOWNLOAD_INDEX],
}


SENTINELSAT_FRP_OPTIONS = {
    # Start, End date, YYYYMMDD, must be string format. The summer period is adviced.
    'date': SENTINELSAT_DATE,
    'platformname': 'Sentinel-3',  # Satellite platform name.
    'producttype': 'SL_2_FRP___',  # Find products @ link below:
    'processinglevel': '2',  # Level of the product.
    'processingmode': 'Offline',  # Processing mode, i.e. Offline, NRT.
    'footprint_file': 'greece.geojson',
    'username': 'connorchewning',
    'password': 'Denmark21',
    'data_dir': 's3_FRP_data',  # Directory to download data to.
    # Must be > 1. Multiple indexes list will download multiple available files.
    'download_index': DOWNLOAD_INDEX,
}

SENTINELSAT_OLCI_OPTIONS = {
    # Start, End date, YYYYMMDD, must be string format. The summer period is adviced.
    'date': SENTINELSAT_DATE,
    'platformname': 'Sentinel-3',  # Satellite platform name.
    'producttype': 'OL_1_EFR___',  # Find products @ link below:
    'processinglevel': '1',  # Level of the product.
    'processingmode': 'Offline',  # Processing mode, i.e. Offline, NRT.
    'footprint_file': 'greece.geojson',
    'username': 'connorchewning',
    'password': 'Denmark21',
    'data_dir': 's3_OLCI_data',  # Directory to download data to.
    # Must be > 1. Multiple indexes list will download multiple available files.
    'download_index': DOWNLOAD_INDEX,
}



# Get S3 Data
# getSentSatData(sentinelsat_options=SENTINELSAT_OPTIONS)
# getSentSatData(sentinelsat_options=SENTINELSAT_FRP_OPTIONS)
# getSentSatData(sentinelsat_options=SENTINELSAT_OLCI_OPTIONS)


# Used to look at files in S3 folders
# lookAtData()

# Display data
# displayData(sentinelsat_options=SENTINELSAT_OPTIONS)
# makeGIF('f1_band', '../figures/gif_files/f1_band', 0.5)

# #####################################################################################
# print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print('Extracting satellite products from downloads:')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
# #####################################################################################


# Get Needed Band Data in usable format
# List of wanted files
FILE_LIST = ['F1_BT_fn.nc',
             'F2_BT_in.nc',
             'S7_BT_in.nc',
             'S8_BT_in.nc',
             'S9_BT_in.nc',
             'geodetic_fn.nc',
             'geodetic_in.nc',
             'S2_radiance_an.nc',
             'S3_radiance_an.nc',
             'S6_radiance_an.nc',
             'geometry_tn.nc', 'geodetic_tx.nc',
             'S2_quality_an.nc', 'S3_quality_an.nc', 'S6_quality_an.nc']
# getProducts(data_dir=SENTINELSAT_OPTIONS['data_dir'], file_list = FILE_LIST)

# Calculate the reflectance from the radiance values
# calcReflectance(data_dir=SENTINELSAT_OPTIONS['data_dir'])


# #####################################################################################
# print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print('Creating Masks:')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
# #####################################################################################


# Compute Initial MODIS Thresholds
INITIAL_THRESHOLDS = {
    'MIR': 310,
    'DIF': 10,
    'DAY_NIGHT_ZENITH': 85
}

# # Mark Which pixels are considered night pixels
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


# #####################################################################################
# print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print('Calculating Fire Pixels:')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
# #####################################################################################


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
# displayFirePixels(data_dir=SENTINELSAT_OPTIONS['data_dir'], coordinates=COORDINATES, show=False, type='potential')
# makeGIF('potential_fire_pixels', '../figures/gif_files/potential_fire_pixels', 0.5)

# Determine what the background values are for each pixel identified as holding a possible fire
# calculateBackground(data_dir=SENTINELSAT_OPTIONS['data_dir'])

# Run the confirmation thresholds to confirm which pixels hold active fires
# confirmFirePixels()
# # makeGIF('confirmed_fire_pixels_full', '../figures/gif_files/confirmed_fire_pixels_full', 0.5)
# displayFirePixels(data_dir=SENTINELSAT_OPTIONS['data_dir'], coordinates=COORDINATES, show=False, type='confirmed')
# makeGIF('confirmed_fire_pixels', '../figures/gif_files/confirmed_fire_pixels', 0.5)

# Run the false alarm detection algorithm to further narrow down results
# detectFalseAlarms()
# # makeGIF('false_alarm_pixels', '../figures/gif_files/false_alarm_pixels', 0.5)
# displayFirePixels(data_dir=SENTINELSAT_OPTIONS['data_dir'], coordinates=COORDINATES, show=False, type='false_alarm')
# makeGIF('confirmed_fire_pixels_FA_removed', '../figures/gif_files/confirmed_fire_pixels_FA_removed', 0.5)




#####################################################################################
# Reduce the resolution of the data for export to biomass program
# print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print('Creating Grid for Resolution Reduction:')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
#####################################################################################

# Determine the resolution of the grid in degrees
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



####################################################################################
print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Running Quantitative Validation with Sentinel-3 Level 2 FRP Product:')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
####################################################################################


lookAtData(data_dir=SENTINELSAT_FRP_OPTIONS['data_dir'], basedir='S3A_SL_2_FRP____20210806T081630_20210806T081930_20210807T162321_0179_075_021_2340_LN2_O_NT_004.SEN3')


FILE_LIST_FRP = ['FRP_in.nc']
getProducts(data_dir=SENTINELSAT_FRP_OPTIONS['data_dir'], file_list = FILE_LIST_FRP)


extractValidationData(data_dir=SENTINELSAT_FRP_OPTIONS['data_dir'])







#  # assign points to grid
# assignGridYN(grid_info=grid_info, data_dir=SENTINELSAT_FRP_OPTIONS['data_dir'])
# makeGIF('grid_location', '../frp_figures/gif_files/grid_location', 0.5)

# # Decrease Resolution based on gridYN
# assignGridID(grid_info=grid_info, data_dir=SENTINELSAT_FRP_OPTIONS['data_dir'])
# makeGIF('grid', '../frp_figures/gif_files/grid', 0.5)

# # Make cell averages
# makeCellAverage(grid_info=grid_info, data_dir=SENTINELSAT_FRP_OPTIONS['data_dir'])

# # Show cell averages
# displayCellValue(grid_info=grid_info, data_dir=SENTINELSAT_FRP_OPTIONS['data_dir'])
# makeGIF('fire_cells', '../frp_figures/gif_files/fire_cells', 0.5)



#####################################################################################
# print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print('Running Visual Validation with Sentinel-3 Level 1B OLCI Product:')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
#####################################################################################




#####################################################################################
print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Script Complete:')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
#####################################################################################