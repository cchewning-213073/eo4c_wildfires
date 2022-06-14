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

SENTINELSAT_DATE = ['20180529', '20180530']

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
computeInitThresh_MODIS(data_dir='s3_data', thresholds={'MIR': 310, 'DIF': 10})

# Display Potential Fire displayFirePixels
displayFirePixels(data_dir='s3_data', coordinates={
                  'lon_min': 22, 'lon_max': 24, 'lat_min': 38, 'lat_max': 40}, show=True)
