#Code for biomass destruction assessment due to fire 
#Course 30574 Earth observations for monitoring changes
#Group Fire

#%% Import needed packages
#import packages that we need
import os
import pdb
from rasterstats import zonal_stats
import rasterio
from rasterio.plot import show
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
import pyproj

import cartopy.crs as ccrs


os.environ['PROJ_LIB'] = os.path.abspath(r'/opt/conda/share/proj')
os.environ['GDAL_DATA'] = os.path.abspath(r'/opt/conda/share')

#%% Set working directory
# Get working directory & set new one
os.getcwd()

# Franca's working directory
os.chdir(r'C:\Users\Franca Bauer\Documents\Danemark\4.Semester\EO4Change')

#%% Define needed inputs

#Open the Biomass Geotiff Map from year 2010 from GlobBiomass mission
#Corresponding file can be downloaded here: https://globbiomass.org/wp-content/uploads/GB_Maps/Globbiomass_global_dataset.html
#Save the downloaded file in a folder "Data" in defined working directory
fname = os.path.join('Data','N40E020_agb.tif')

#Define area of interest (here: around Greece)
upper_left_x = 22
upper_left_y = 40
lower_right_x = 24
lower_right_y = 38

#Get active fire map
ac_fire = 
#%% Here beginning agb processing 

#Could plot the geotiff file here (but it is big and takes quite some time)
#img = rasterio.open(fname)
#show(img)

#Open the geotiff file with agb for Above Ground Biomass
raster_agb = gdal.Open(fname)

#Get extent of geotiff file (pixel size, longitude and latitude, pixel amount per side)
xmin, xpixel, _, ymax, _, ypixel = raster_agb.GetGeoTransform()
width, height = raster_agb.RasterXSize, raster_agb.RasterYSize
xmax = xmin + width * xpixel
ymin = ymax + height * ypixel

#Clip the geotiff to the extent of area of interest
window = (upper_left_x, upper_left_y, lower_right_x, lower_right_y)
gdal.Translate('greece_agb.tif', fname, projWin = window)

#Plot agb in Greece (new extent)
raster_agb_greece = rasterio.open('greece_agb.tif')
show(raster_agb_greece)

#Open new geotiff of greece with gdal
raster_agb_greece_fine_res = gdal.Open('greece_agb.tif')

#Change pixel resolution of geotiff from  0.0008888 to 0.05 pixel resolution
#'tr' = target file's resolution & 'r' = sampeling algorithem (here: moving average)
options = gdal.WarpOptions(options=['tr','r'], xRes=0.05, yRes=0.05, resampleAlg= gdal.GRA_Average)
raster_agb_greece_coarse_res = gdal.Warp('greece_agb_0.05res.tif', raster_agb_greece_fine_res, options=options)

#Check of transformation has worked
newprops = raster_agb_greece_coarse_res.GetGeoTransform()
print('new pixel xsize:', newprops[1], 'new pixel ysize:', -newprops[-1])

#Read coarser geotiff of Greece in as np.array
agb_ar = raster_agb_greece_coarse_res.ReadAsArray()

#Plot the agb in greece with new resolution
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('Greens')
img = ax.imshow(agb_ar, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap=cmap,)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Biomass [Mg/ha]')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()

#%% Calculate the area of each pixel to get biomass per pixel and not per hectar
#agb is in [Mg/ha] = [t/ha] 

#Degrees to meter
import haversine as hs

loc1=(upper_left_x, lower_right_y)
loc2=(upper_left_x, upper_left_y)
distance = hs.haversine(loc1,loc2) #Distance in km for whole interest area in longitude

loc1=(upper_left_x, lower_right_y)
loc2=(lower_right_x, lower_right_y)
distance2 = hs.haversine(loc1,loc2) #Distance in km for whole interest area in latitude

dis_per_pixel = distance/len(agb_ar) #Distance in km for pixel in longitude
dis2_per_pixel = distance2/len(agb_ar) #Distance in km for pixel in latitude

area_per_pixel_km = dis_per_pixel*dis2_per_pixel #Area of pixel in km2
area_per_pixel_ha = area_per_pixel_km * 100 #Area of pixel in ha

agb_t = np.multiply(agb_ar,area_per_pixel_ha)
agb_kt = np.multiply(agb_t,0.001) #kiloton [kt]

#Plot the agb in greece with new resolution
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('Greens')
img = ax.imshow(agb_kt, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap= cmap)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Biomass [kt] (kiloton)')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()


#%% Convert biomass map to release carbon map

#Source for calculation:
#https://www3.epa.gov/ttnchie1/conference/ei11/pm/trozzi.pdf

#with assumption for complete fire = complete destruction to calculate the potential carbon release
degree_fire = 1 #Value between 0-1: describing the degree of destruction
burnt_agb = np.multiply(agb_kt, degree_fire)

#Plot the agb in greece with new resolution
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('YlOrBr')
img = ax.imshow(burnt_agb, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap= cmap)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Potentially burnt Biomass [kt] (kiloton)')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()

# carbon's quantity contained in the biomass is 0.45%
beta = 0.45
pot_released_carbon = np.multiply(burnt_agb, beta)

#Plot the agb in greece with new resolution with mass per pixel
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('OrRd')
img = ax.imshow(pot_released_carbon, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap= cmap)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Potentially released carbon [kt] (kiloton)')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()

#%% Convert released carbon map to released C02, CH4, CO maps

pot_released_CO2 = np.multiply(pot_released_carbon, 0.888)
pot_released_CO2 = np.multiply(pot_released_CO2, 44/12)
pot_released_CO = np.multiply(pot_released_carbon, 0.1)
pot_released_CO = np.multiply(pot_released_CO, 28/12)
pot_released_CH4 = np.multiply(pot_released_carbon, 0.012)
pot_released_CH4 = np.multiply(pot_released_CH4, 16/12)

#Plot the potential CO relased in greece with new resolution with mass per pixel
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('Blues')
img = ax.imshow(pot_released_CO, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap= cmap)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Potentially released CO [kt] (kiloton)')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()

#Plot the potential CO2 relased in greece with new resolution with mass per pixel
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('Reds')
img = ax.imshow(pot_released_CO2, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap= cmap)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Potentially released CO2 [kt] (kiloton)')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()

#Plot the potential CH4 relased in greece with new resolution with mass per pixel
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('Purples')
img = ax.imshow(pot_released_CH4, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap= cmap)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Potentially released CH4 [kt] (kiloton)')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()

#%% Preperation code for active fire map

burnt_biomass = np.zeros(shape=(40,40))

for i in range(len(agb_kt)):
    for j in range(len(agb_kt[i])):
        if ac_fire[i][j] == 1:
            burnt_biomass[i][j] = agb_kt[i][j]
        elif ac_fire[i][j] == 0:
            burnt_biomass[i][j] = 0
        else:
            print("Value is not known:", new_ar[i][j])
 
#Plot the burnt biomass
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [22.00, 24.00, 38.00, 40.00]
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
cmap = cm.get_cmap('YlOrBr')
img = ax.imshow(burnt_biomass, extent=extent, origin='upper', transform=ccrs.PlateCarree(), alpha=0.25, cmap= cmap)  # make this layer on top
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title('Potentially released CH4 [kt] (kiloton)')
plt.xlabel('Degree E')
plt.ylabel('Degree N')
plt.colorbar(img)
plt.show()

#%% Preperation calculation for total carbon emission

burnt_biomass_lst =  burnt_biomass.flatten().tolist()

sum_biomass = sum(burnt_biomass_lst)
sum_carbon = sum_biomass * beta
sum_CO2 = sum_carbon*0.888*(44/12)
sum_CO = sum_carbon*0.1*(28/12)
sum_CH4 = sum_carbon*0.012*(16/12)


