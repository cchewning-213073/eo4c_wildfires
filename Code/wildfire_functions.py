import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import glob
from netCDF4 import Dataset


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
                        processinglevel=sentinelsat_options['processinglevel'],
                        # processingmode=sentinelsat_options['processingmode']
                         )

    # convert to Pandas DataFrame
    print('Retrieving Products.')
    products_df = api.to_dataframe(products)

    if len(products_df) < 1:
        exit('No products available.')

    print(f"Number of products available: {len(products_df)}")
    cwd = os.getcwd()  # remember current work directory (CWD).
    os.chdir(sentinelsat_options['data_dir'])  # Change directory.
    for index in sentinelsat_options['download_index']:
        print(f"Downloading product: {products_df.head(index)['title']}")

        # Select and download product to data directory.
        product_df = products_df.head(index)  # Get desired product based on selected index.
        # Download product. Needs '.index' as it cannot download df directly.
        api.download_all(product_df.index)

    os.chdir(cwd)  # Return to project directory.


"""
Get Sentinel Data
"""
def displayData(sentinelsat_options: dict):

    os.chdir('s3_data')

    # # move into data dir
    basedir = 'S3A_SL_1_RBT____20180529T212058_20180529T212358_20201230T052814_0179_031_357_0720_LR1_R_NT_004.SEN3'

    # File name (include extenstion) -> and move into this
    files = glob.glob(basedir+os.sep+'S7_BT_*i*.nc')
    for ff in files:
        print(ff)    

    file = files[0]

    print('\n\n')

    with Dataset(file) as src: 
    
        # Each datafile contains multiple bands
        for band, variable in src.variables.items():

            print('\n------' + band + '------')
            
            if "_BT_in" in band:
                bandName = band 
                # Printing the attributes of the BT_in variables:
                for attrname in variable.ncattrs():
                    print("{} -- {}".format(attrname, getattr(variable, attrname))) 


    data = Dataset(file)
    data.set_auto_mask(False) #If you want to work directly with arrays -> says we dont want to know what is flagged and not. Contains inherit information about the quality and it leads to many NA values.
    bandData = data.variables[bandName][:]
    data.close()


    plt.figure(figsize=(15,8))
    plt.imshow(bandData, vmin=220, vmax=280)
    plt.colorbar()
    plt.show()

    # file = os.path.join(folder, file)

    # # Name of the product in the level-2 file.
    # s5p_product_name = 'F1 BT NADIR'
    # # Opens the .nc file.
    # s5p_product = xr.open_dataset(sentinelsat_options['data_dir'] + '/' + file, group='PRODUCT')

    # s5p_fig = plt.figure()

    # # Make the figure have lat lon projection.
    # ax = plt.axes(projection=ccrs.PlateCarree())

    # # Set the extent
    # ax.set_extent([4, 16, 52, 59], ccrs.PlateCarree())

    # # Apply coastlines. Resolution can be 10m, 50m or 110m. Higher resolution may be slower.
    # ax.coastlines(resolution='10m', color='black', linewidth=1)

    
    # # Plot the product as a scatter plot. Note that x=lon and y=lat.
    # plt.scatter(x=s5p_product['longitude'], y=s5p_product['latitude'],
    #             c=s5p_product[s5p_product_name].squeeze(),
    #             vmin=np.nanquantile(s5p_product[s5p_product_name], q=0.01),
    #             vmax=np.nanquantile(s5p_product[s5p_product_name], q=0.99))

    # plt.title(s5p_product_name)


    # # Get colorbar. You can add colorbar units with cbar.set_label('unit').
    # # The unit can be found by s5p_product['your product'].attrs['units'].
    # cbar = plt.colorbar()

    # cbar.set_label(s5p_product[s5p_product_name].attrs['units'])

    # # The figure can be saved with s5p_fig.savefig('cool_s5p_fig.png', format='png').
    # s5p_fig.savefig('initial_s5p_fig.png', format='png')

    # # The plot will be shown until the window is closed.
    # plt.show()
