# This code fills small gaps with linear interpolation 
# This code can model lake surface area snow cover based on snowcover of surrounding land
# This code can model glacier surface area snow cover based on elevation, aspect, and slope (ML model)
# Outputs:
# Gap-filled snowdepth maps for each phase

# Import packages
import pyrsgis
import numpy as np
import pandas as pd
import os
import geopandas
import rasterio
import rasterio.fill
from rasterio import features
from rasterstats import zonal_stats
from pathlib import Path

# Import input data -----------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
date = var['date'][0]
BEversion = var['BEversion'][0]
lakemodel = var['lakemodel'][0]
frac = var['frac'][0]
phases = []
x = var['phases'][var['phases'].notna()]
for n in range(len(x)):
    a = x[n]
    phases.append(a)

# Import bare earth
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan
del nans

# Import extent mask without lakes.
[R,WS_gaps]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_no_lakes_'+str(resolution1)+'m.tif'))
nans=np.where(WS_gaps<=0)
WS_gaps[nans]=np.nan

# Read lakes vector dataset and create 100m buffer
lakes = geopandas.read_file(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/vector/'+str(watershed)+'_lakes/')
lakes['buffered'] = lakes.buffer(distance=100)

for n in range(len(phases)):
    phase=phases[n]
    # Load Snowdepth raster
    SD_in = rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phase)+'_capped_clipped_vegcorrected_'+str(resolution1)+'m.tif')

    # Calculate mean, median, min, max snowdepth around each lake
    stats = zonal_stats(lakes['buffered'], SD_in.read(1), affine=SD_in.transform, nodata = SD_in.nodata, stats=["mean", "median", "max", "min"])
    lakes_joined = lakes.join(geopandas.GeoDataFrame(stats))

    # Calculate fraction of snow on lake compared to snow on land (based on median value)
    lakes_joined['snowdepth'] = lakes_joined['median'] * frac

    # Rasterize lake polygons with snowdepth value
    geom = [shapes for shapes in lakes_joined['geometry'].geometry]
    geom_value = ((geom,value) for geom, value in zip(lakes_joined.geometry, lakes_joined['snowdepth']))
    rasterized = features.rasterize(geom_value,
                                    out_shape = SD_in.shape,
                                    fill = np.nan,
                                    out = None,
                                    transform = SD_in.transform,
                                    all_touched = False)
    with rasterio.open(
            str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/lakes_temp.tif', "w",
            driver = "GTiff",
            crs = SD_in.crs,
            transform = SD_in.transform,
            dtype = rasterio.float32,
            count = 1,
            width = SD_in.width,
            height = SD_in.height) as dst:
        dst.write(rasterized, indexes = 1)
    del SD_in,geom,geom_value,rasterized,dst,stats

    # Reload snowdepth (land) and snowdepth (lake) rasters and merge them
    [R,land]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phase)+'_capped_clipped_vegcorrected_'+str(resolution1)+'m.tif'))
    [S,lake]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/lakes_temp.tif'))

    nans=np.where(land<-100)
    land[nans]=np.nan
    land_flattened=np.ndarray.flatten(land)
    lake_flattened=np.ndarray.flatten(lake)
    notnans = ~np.isnan(lake_flattened)
    j=np.ndarray.flatten(np.argwhere(notnans))
    only_lakes=lake_flattened[notnans]
    land_flattened[j]=only_lakes

    # Reshape results
    dims=np.shape(land)
    SD_out=np.reshape(land_flattened,(dims[0],dims[1]))

    # Optional: Import manual corrections masks for noisy areas that were manually outlined (in GIS software) after running the '2_SnowDepth_corrections.py' script
    file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Manual_corrections/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(phase)+'_manual_corrections.tif')
    if file.is_file():
        [R,mask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Manual_corrections/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(phase)+'_manual_corrections.tif', bands='all'))
        mask[mask==0]= np.nan
        i=np.where(mask==1)
        SD_out[i]= np.nan

    # Determine areas where interpolation is required (i.e. within the boundary of the watershed, not in lakes or glaciers) and set these areas to 0 and everything else to 1
    SD_flattened=np.ndarray.flatten(SD_out)
    interpolation_areas=(SD_flattened/SD_flattened).astype('float64')
    b=np.ndarray.flatten(np.argwhere(np.isnan(SD_flattened))).astype('int64')
    interpolation_areas[b]=0
    nans=np.argwhere(np.isnan(interpolation_areas))
    interpolation_areas[nans]=1
    dims=np.shape(SD_out)
    interpolation_areas_2d=np.reshape(interpolation_areas,(dims[0],dims[1]))
    del b,nans,dims,SD_flattened,interpolation_areas
        
    filled=rasterio.fill.fillnodata(SD_out,interpolation_areas_2d) #enter 'interpolation_areas_2d' if using mask

    if lakemodel == 'N':
        # Remove lakes
        [R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_lakes_'+str(resolution1)+'m.tif', bands='all'))
        lakemask[lakemask<=0]= np.nan
        i=np.where(lakemask==1)
        filled[i]= np.nan

    # Output ---------------------------------------------------------------------------------------
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m')
    pyrsgis.export(filled,R,filename='Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phase)+'_capped_clipped_vegcorrected_filled_lakemodel'+str(lakemodel)+'_'+str(resolution1)+'m.tif') 
    del filled    
    print('Phase '+str(n+1)+'/'+str(len(phases))+' complete')

# Save processing variables
var.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Metadata/'+str(extent)+'_'+str(year)+'_processing_variables_'+str(date)+'.csv', index = False)
