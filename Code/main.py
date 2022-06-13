"""
    30574 Earth observations for monitoring changes (EO4Change) Wildfire and Biomass Impacts.
"""


##### Import needed packages and external files
import os
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt


##### Define Data product Dictionaries

SENTINELSAT_DATE = ['20210529', '20210530']



SENTINELSAT_OPTIONS = {
    # Start, End date, YYYYMMDD, must be string format. The summer period is adviced.
    'date': SENTINELSAT_DATE,
    'platformname': 'Sentinel-3',  # Satellite platform name.
    'producttype': 'L2__NO2___',  # Find products @ link below:
    # https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-5p/products-algorithms
    'processinglevel': 'L2',  # Level of the product.
    'processingmode': 'Offline',  # Processing mode, i.e. Offline, NRT.
    # name of geojson file. Inset filename from directory or make your own https://geojson.io
    'footprint_file': 'map.geojson',
    'username': 's5pguest',  # Default username for Sentinel-5P download, no need to sign-up.
    'password': 's5pguest',  # Default password for Sentinel-5P download, no need to sign-up.
    'data_dir': 's5p_data',  # Directory to download data to.
    # Must be > 1. Multiple indexes list will download multiple available files.
    'download_index': [1,2,3,4,5],
}