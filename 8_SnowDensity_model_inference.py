# This script calculates snowdensity with a machine learning random forest model based on
# in-situ snowdensity & snowdepth, elevation (snow-free bare earth), slope, northness, eastness, curvature, canopy cover, canopy height, and Xt. 
# This script outputs:
# A snowdensity raster map

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='CRU' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2024' # Enter year of interest
phase='P1' # Enter survey phase. NOTE only one phase can be run at the time in this script
BEversion = 2 # Enter Bare Earth version number
resolution = 2 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
lakemodel = 'Y' # Enter 'Y' or 'N' for including modelled SnowDepth and SnowDensity on lakes

import numpy as np
import pyrsgis
import os
import joblib
import pandas as pd
import geopandas
import gc
import rasterio
from rasterio import features
from rasterstats import zonal_stats
from pathlib import Path

# Import data -----------------------------------------------------------------------------------------------------
# Import snow density model
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Overall_density_model/All_parameters')
rf = joblib.load('RF_density_model_'+str(watershed)+'.joblib')
all_scalers=[]
for n in range(1,10):
    scaler = joblib.load('scaler'+str(n)+'.pkl')
    all_scalers.append(scaler)
del scaler

# Import model input data (bare earth - derived and Canopy data):
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m')
DEMFiles=[str(subbasin)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif']
input=[]
for n in range(len(DEMFiles)):
    [R,x]=np.array(pyrsgis.raster.read(DEMFiles[n], bands='all'))
    input.append(x)
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/resolution_'+str(resolution)+'m')
CanopyFiles=[str(subbasin)+'_CC_'+str(resolution)+'m.tif',str(subbasin)+'_CH_'+str(resolution)+'m.tif']
for n in range(len(CanopyFiles)):
    [R,x]=np.array(pyrsgis.raster.read(CanopyFiles[n], bands='all'))
    input.append(x)
rows=x.shape[0]
cols=x.shape[1]
# Set nan values
for n in range(len(input)):
    x=input[n]
    x[x<-1]=np.nan
    input[n]=x
del DEMFiles,CanopyFiles,x,n

# Import model input data (Xt data):
[R,Xt]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution)+'m/Distributed_Xt_'+str(subbasin)+'_'+str(year)+'_'+str(phase)+'.tif', bands='all'))
for n in range(len(Xt)):
    x=Xt[n]
    x[x<0]=np.nan
    Xt[n]=x
Xt=np.ndarray.flatten(Xt)
del x,n

# Import model input data (snow depth data converted to cm):
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/')
[R,SD]=np.array(pyrsgis.raster.read(str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif', bands='all'))
nans=np.where(SD<-100)
SD = SD.astype(float)        
SD[nans] = np.nan          
SD=np.ndarray.flatten(SD)*100 # convert to cm

# Import snow free areas. A raster mask is used if available, otherwise an elevation threshold.
file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SFA.tif')
if file.is_file():
    [R,SFA]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SFA.tif', bands='all'))
else:
    x=pd.read_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/SFETs_'+str(subbasin)+'_'+str(year)+'.csv')
    SFET=x[x['Survey']==phase]
    SFET=SFET['SFETs'].iloc[0]
    BE=input[3]
    SFA=BE/BE
    i=np.where(BE<SFET)
    j=np.where(BE>=SFET)
    SFA[i]=1
    SFA[j]=0

# Set NaNs at SFA for all input data
x=SD
y=np.ndarray.flatten(SFA)
nans=np.where(y==1)
x[nans]=np.nan
SD=x

x=Xt
y=np.ndarray.flatten(SFA)
nans=np.where(y==1)
x[nans]=np.nan
x=x.reshape(len(x),1)
Xt=x

Surface_Characteristics=[]
for n in range(len(input)):
    x=input[n]
    x=np.ndarray.flatten(x)
    y=np.ndarray.flatten(SFA)
    nans=np.where(y==1)
    x[nans]=np.nan
    x=x.reshape(len(x),1)
    Surface_Characteristics.append(x)
del n,x,y

# Organise input data for each survey
input_parameters=[SD.astype('float64'),Surface_Characteristics[0].astype('float64'),Surface_Characteristics[5].astype('float64'),Surface_Characteristics[6].astype('float64'),Surface_Characteristics[3].astype('float64'),Surface_Characteristics[4].astype('float64'),Surface_Characteristics[1].astype('float64'),Xt.astype('float64'),Surface_Characteristics[2].astype('float64')]
del Surface_Characteristics,Xt

# Normalise input variables
for n in range(len(input_parameters)):   
    x=input_parameters[n]
    x=x.reshape(len(x),1)
    scaler = all_scalers[n]
    X_normalized = scaler.transform(x)
    X_normalized=X_normalized.reshape(len(X_normalized),)
    x=X_normalized
    input_parameters[n]=x  
del x,scaler,all_scalers,X_normalized

# Set nans to 0 in input parameters, and get indices of nans
all_nans_all_surveys=[]
for n in range(len(input_parameters)):
    ip=input_parameters[n]
    all_nans = np.nonzero(np.isnan(ip))
    ip[all_nans]=0
    all_nans_all_surveys.append(all_nans)
    input_parameters[n]=ip
del n,ip,nans,all_nans   

# Reformat model input data
for n in range(len(input_parameters)):
    x=input_parameters[n]
    y=x.reshape(len(x),1)
    input_parameters[n]=y
input_parameters=np.concatenate(input_parameters,axis=1)
del n,x,y
gc.collect()

# Run snow density model
Simulated_density=rf.predict(input_parameters)
del input_parameters

# OPTIONAL adjust density values
Simulated_density=Simulated_density + 0.071
# OPTIONAL adjust density values
#os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/')
#[R,SD]=np.array(pyrsgis.raster.read(str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif', bands='all'))
#x=SD
#y=-0.0137*x+0.3327
#y=np.ndarray.flatten(y)
#z=np.ndarray.flatten(SFA)
#nans=np.where(z==1)
#y[nans]=np.nan
#Simulated_density=y

# Create mask for extent of snowdepth data
study_area=SD
i=np.where(SD<0)
j=np.where(SD>=0)
study_area[i]=np.nan
study_area[j]=1

# Set the pixels below snow cover and outside the data outline to nans and reshape into 2-dimensional arrays
Simulated_density=Simulated_density*study_area
Simulated_density=np.reshape(Simulated_density,(rows,cols))
del all_nans_all_surveys,rows,cols

# Remove lakes
[R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_lakes_'+str(resolution)+'m.tif', bands='all'))
i=np.where(lakemask==1)
Simulated_density[i]= np.nan

# Export simulated snow density map
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/')
pyrsgis.raster.export(Simulated_density, R, filename=str(watershed)+'_'+str(year)+'_'+str(phase)+'_SnowDensity_lakemodel'+str(lakemodel)+'.tif')

if lakemodel == 'Y':
    # Read lakes vector dataset and create 100m buffer
    lakes = geopandas.read_file(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/vector/'+str(watershed)+'_lakes/')
    lakes['buffered'] = lakes.buffer(distance=100)

    # Load Density raster
    SD_in = rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phase)+'_SnowDensity_lakemodel'+str(lakemodel)+'.tif')

    # Calculate mean, median, min, max snowdensity around each lake
    stats = zonal_stats(lakes['buffered'], SD_in.read(1), affine=SD_in.transform, nodata = SD_in.nodata, stats=["mean", "median", "max", "min"])
    lakes_joined = lakes.join(geopandas.GeoDataFrame(stats))

    # Calculate snowdensity on lake compared to snow on land (based on median value)
    lakes_joined['snowdensity'] = lakes_joined['median']

    # Rasterize lake polygons with snowdensity value
    geom = [shapes for shapes in lakes_joined['geometry'].geometry]
    geom_value = ((geom,value) for geom, value in zip(lakes_joined.geometry, lakes_joined['snowdensity']))
    rasterized = features.rasterize(geom_value,
                                    out_shape = SD_in.shape,
                                    fill = np.nan,
                                    out = None,
                                    transform = SD_in.transform,
                                    all_touched = False)
    with rasterio.open(
            str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/lakes_temp.tif', "w",
            driver = "GTiff",
            crs = SD_in.crs,
            transform = SD_in.transform,
            dtype = rasterio.float32,
            count = 1,
            width = SD_in.width,
            height = SD_in.height) as dst:
        dst.write(rasterized, indexes = 1)
    del SD_in,geom,geom_value,rasterized,dst,stats

    # Reload snowdensity (land) and snowdensity (lake) rasters and merge them
    [R,land]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phase)+'_SnowDensity_lakemodel'+str(lakemodel)+'.tif'))
    [S,lake]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/lakes_temp.tif'))

    nans=np.where(land<-100)
    land[nans]=np.nan
    land_flattened=np.ndarray.flatten(land)
    lake_flattened=np.ndarray.flatten(lake)
    notnans = ~np.isnan(lake_flattened)
    j=np.ndarray.flatten(np.argwhere(notnans))
    only_lakes=lake_flattened[notnans]
    land_flattened[j]=only_lakes

    # Set NaNs at SFA
    x=land_flattened
    y=np.ndarray.flatten(SFA)
    nans=np.where(y==1)
    x[nans]=np.nan
    land_flattened=x

    # Reshape results
    dims=np.shape(land)
    SD_out=np.reshape(land_flattened,(dims[0],dims[1]))

    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m')
    pyrsgis.raster.export(SD_out, R, filename=str(watershed)+'_'+str(year)+'_'+str(phase)+'_SnowDensity_lakemodel'+str(lakemodel)+'.tif')
