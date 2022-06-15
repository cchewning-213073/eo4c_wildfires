"""
    30574 Earth observations for monitoring changes (EO4Change) Wildfire and Biomass Impacts.
"""

import glob
from algorithm_functions import *
from display_functions import *
from prep_functions import *
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import os
print('\n\n\n\n\n')

# Import needed packages and external files


# Define Data product Dictionaries

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

# Get S3 Data
# getSentSatData(sentinelsat_options=SENTINELSAT_OPTIONS)

# Look at data
# displayData(sentinelsat_options=SENTINELSAT_OPTIONS)

# Get Needed Band Data in usable format
# getThresholdProducts(data_dir='s3_data')

# Compute Initial MODIS Thresholds
INITIAL_THRESHOLDS = {
    'MIR': 310,
    'DIF': 10
}
# computeInitThresh_MODIS(data_dir=SENTINELSAT_OPTIONS['data_dir'], thresholds=INITIAL_THRESHOLDS)

# Display Potential Fire displayFirePixels
COORDINATES = {
    'lon_min': 22,
    'lon_max': 24,
    'lat_min': 38,
    'lat_max': 40
}
# displayFirePixels(data_dir=SENTINELSAT_OPTIONS['data_dir'], coordinates=COORDINATES, show=True)

# Create Grid
RESOLUTION = 0.05
grid_info = createGrid(res=RESOLUTION, coordinates=COORDINATES)

# #  assign points to grid
# assignGridYN(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])
#
# # Decrease Resolution based on gridYN
# assignGridID(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])
#
# # Make cell averages
# makeCellAverage(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])

# Show cell averages
displayCellValue(grid_info=grid_info, data_dir=SENTINELSAT_OPTIONS['data_dir'])
