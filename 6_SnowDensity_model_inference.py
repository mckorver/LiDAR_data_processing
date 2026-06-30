# This script calculates snowdensity with a machine learning random forest model based on
# in-situ snowdensity & snowdepth, elevation (snow-free bare earth), slope, northness, eastness, curvature, canopy cover, canopy height, and Xt. 
# This script outputs:
# A snowdensity raster map

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
# Import processing variables
var = pd.read_csv('V:/LiDAR_data_processing/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str, 'DENSversion':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution2 = var['resolution2'][0]
date = var['date'][0]
BEversion = var['BEversion'][0]
CANversion = var['CANversion'][0]
lakemodel = var['lakemodel'][0]
DENSversion = var['DENSversion'][0]
bias_correction_dens = []
phases=[]
days_in_season=[]
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if b == 'days_in_season':
            a.append(int(y))
        else:
            a.append(y)
append_fun(bias_correction_dens,'bias_correction_dens')
append_fun(phases,'phases')
append_fun(days_in_season,'days_in_season')

# Import snow density model
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
model_var = pd.read_csv(str(watershed)+'_ML_model_processing_variables_v'+str(DENSversion)+'.csv')
predictors=[]
x = model_var['predictors'][model_var['predictors'].notna()]
for n in range(len(x)):
    a = x[n]
    predictors.append(a)
rf = joblib.load('RF_density_model_'+str(watershed)+'_v'+str(DENSversion)+'.joblib')
all_scalers=[]
for n in range(1,len(predictors)+1):
    scaler = joblib.load('scaler'+str(n)+'.pkl')
    all_scalers.append(scaler)
del scaler

for n in range(len(phases)):
    # Import predictor data
    input=[]
    for m in predictors:
        if m == "snow_depth_m":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif', bands='all'))  
            x=x*100 # convert to cm
            input.append(x)
            rows=x.shape[0]
            cols=x.shape[1]
        elif m == "elevation_lidar":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
            input.append(x)
        elif m == "slope_lidar":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
            input.append(x)
        elif m == "curvature_lidar":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
            input.append(x)
        elif m == "eastness_lidar":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
            input.append(x)
        elif m == "northness_lidar":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
            input.append(x)
        elif m == "Xt_model":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution2)+'m/Distributed_Xt_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'.tif', bands='all'))
            input.append(x)
        elif m == "canopy_cover_lidar":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_CC_v'+str(CANversion)+'_'+str(resolution2)+'m.tif', bands='all'))
            input.append(x)
        elif m == "canopy_height_lidar":
            [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_CH_v'+str(CANversion)+'_'+str(resolution2)+'m.tif', bands='all'))
            input.append(x)
        elif m == "day_in_season":
            x=np.full((rows,cols),days_in_season[n])
            input.append(x)
    for m in range(len(input)):
        x=input[m]
        x=x.astype('float64')
        x=np.ndarray.flatten(x)
        x[x<-1]=np.nan
        input[m]=x
    del x,m

    # We're using the snow depth map later so saving it here seperately
    index = predictors.index('snow_depth_m')
    SD=input[index]

    # Import snow free areas. A raster mask is used if available, otherwise an elevation threshold.
    file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif')
    if file.is_file():
        [R,SFA]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif', bands='all'))
    else:
        x=pd.read_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/SFETs_'+str(extent)+'_'+str(year)+'.csv')
        SFET=x[x['Survey']==phases[n]]
        SFET=SFET['SFETs'].iloc[0]
        BE=input[1]
        SFA=BE/BE
        i=np.where(BE<SFET)
        j=np.where(BE>=SFET)
        SFA[i]=1
        SFA[j]=0

    # Set all input data to NAN in the snowfree areas, and reshape the data arrays
    for m in range(len(input)):
        x=input[m]
        y=np.ndarray.flatten(SFA)
        nans=np.where(y==1)
        x[nans]=np.nan
        x=x.reshape(len(x),1)
        input[m]=x
    del m,x,y

    # Normalise input variables
    for m in range(len(input)):   
        x=input[m]
        scaler = all_scalers[m]
        X_normalized = scaler.transform(x)
        X_normalized=X_normalized.reshape(len(X_normalized),)
        x=X_normalized
        input[m]=x  
    del x,scaler,X_normalized

    # Set nans to 0 in input parameters, and get indices of nans
    for m in range(len(input)):
        ip=input[m]
        all_nans = np.nonzero(np.isnan(ip))
        ip[all_nans]=0
        input[m]=ip
    del m,ip,nans,all_nans   

    # Reformat model input data
    for m in range(len(input)):
        x=input[m]
        y=x.reshape(len(x),1)
        input[m]=y
    input_reshaped=np.concatenate(input,axis=1)
    del m,x,y
    gc.collect()

    # Run snow density model
    Simulated_density=rf.predict(input_reshaped)

    # Create mask for extent of snowdepth data and Set the pixels below snow cover and outside the data outline to nans
    SD_mask=SD.copy()
    j=np.where(SD>=0)
    SD_mask[j]=1
    Simulated_density=Simulated_density*SD_mask

    # OPTIONAL adjust density values
    Simulated_density=Simulated_density + bias_correction_dens[n]
    # OPTIONAL adjust density values with an equation based on snow depth
    #y=-0.142*(SD/100)+0.2015
    #y[y<0]=0
    #Simulated_density=Simulated_density-y

    # reshape into 2-dimensional arrays
    Simulated_density=np.reshape(Simulated_density,(rows,cols))
    del rows,cols

    # Remove lakes
    [R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution2)+'m/'+str(extent)+'_lakes_'+str(resolution2)+'m.tif', bands='all'))
    i=np.where(lakemask==1)
    Simulated_density[i]= np.nan

    # Export simulated snow density map
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/')
    pyrsgis.raster.export(Simulated_density, R, filename=str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif')

    if lakemodel == 'Y':
        # Read lakes vector dataset and create 100m buffer
        lakes = geopandas.read_file(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution2)+'m/vector/'+str(watershed)+'_lakes/')
        lakes['buffered'] = lakes.buffer(distance=100)

        # Load Density raster
        SD_in = rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif')

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
                str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/lakes_temp.tif', "w",
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
        [R,land]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif'))
        [S,lake]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/lakes_temp.tif'))

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

        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m')
        pyrsgis.raster.export(SD_out, R, filename=str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif')
    del SFA,input,SD,Simulated_density
    print('Phase '+str(n+1)+'/'+str(len(phases))+' complete')

# Save processing variables
var.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Metadata/'+str(extent)+'_'+str(year)+'_processing_variables_'+str(date)+'.csv', index = False)
