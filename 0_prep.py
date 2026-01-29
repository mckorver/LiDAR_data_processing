# Downsample, clip, or set extents of rasters
# This script is NOT meant to be run in its entirety, only selected code chunks when needed.

# Watershed and subbasin names for reference
# 'MV','SEY','CAP','BurwellLake','LochLomond','PalisadeLake','UpperSeymour'
# 'ENG','Arrowsmith','Fishtail','Cokely'
# 'CRU','Comox','Eric','Moat','Rees','Residual'

watershed='CRU' # ENG, MV, TSI, CRU
subbasin=['CRU'] #Enter subbasin of interest. Repeat watershed acronym if running entire watershed
year='2024'
phases=['P1'] # Survey number
BEversion = 2 # Bare Earth version number
resolution = 1 # Raster resolution in meters
resolution2 = 2 # Raster resolution in meters you want to downscale to
drive = 'K' # File path drive letter
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
lakemodel = 'Y' # Enter 'Y' or 'N' for including modelled SnowDepth on lakes
lakes = '' # Enter '_no_lakes' for watershed mask with lakes cut out or '' for including lakes
glaciers = '' # Enter '_no_glaciers' for watershed mask with glaciers cut out or '' for including glaciers

from osgeo import gdal
import os
import subprocess
import numpy as np
import pyrsgis
import gc
import rasterio

# region # DOWNSAMPLING -----------------------------------------------------------
# NOTE continuous data are resampled with averaging, binary data with nearest neighbor. All nodata values are set to -9999
# Downsample snowdepth to 2 m
for n in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff '+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution2)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution2)+'m/')
    [R,SD]=np.array(pyrsgis.raster.read(str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif', bands='all'))
    pyrsgis.raster.export(SD, R, filename=str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif')

# Downsample Xt to 2 m
for m in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution)+'m')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff Distributed_Xt_'+str(watershed)+'_'+str(year)+'_'+str(phases[m])+'.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution2)+'m/Distributed_Xt_'+str(watershed)+'_'+str(year)+'_'+str(phases[m])+'.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution2)+'m')
    [R,y]=np.array(pyrsgis.raster.read('Distributed_Xt_'+str(watershed)+'_'+str(year)+'_'+str(phases[m])+'.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename='Distributed_Xt_'+str(watershed)+'_'+str(year)+'_'+str(phases[m])+'.tif')

# Downsample DEM model input data to 2 m
inputs = ['_Slope','_Eastness','_Northness','','_Curvature','_Aspect'] # Input model data. The empy '' is for the bare earth file
for m in range(len(inputs)):
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff '+str(watershed)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif '+str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(watershed)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(watershed)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(watershed)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif')

# Downsample canopy model input data to 2 m
inputs = ['_CC','_CH', '_CD'] # Input model data.
for m in range(len(inputs)):
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/resolution_'+str(resolution)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff '+str(watershed)+str(inputs[m])+'_'+str(resolution)+'m.tif '+str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/resolution_'+str(resolution2)+'m/'+str(watershed)+str(inputs[m])+'_'+str(resolution2)+'m.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(watershed)+str(inputs[m])+'_'+str(resolution2)+'m.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(watershed)+str(inputs[m])+'_'+str(resolution2)+'m.tif')

# Downsample lakes to 2 m
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/')
cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(watershed)+'_lakes_'+str(resolution)+'m.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution2)+'m/'+str(watershed)+'_lakes_'+str(resolution2)+'m.tif'
subprocess.run([x for x in cmd.split(" ") if x != ""])

# Downsample glaciers to 2 m
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/')
cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(watershed)+'_glaciers_'+str(resolution)+'m.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution2)+'m/'+str(watershed)+'_glaciers_'+str(resolution2)+'m.tif'
subprocess.run([x for x in cmd.split(" ") if x != ""])

# Downsample Watershed Mask to 2 m
for x in range(len(subbasin)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(subbasin[x])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution)+'m.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution2)+'m/'+str(subbasin[x])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution2)+'m.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(subbasin[x])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution2)+'m.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(subbasin[x])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution2)+'m.tif')

# Downsample SFA to 2 m
for n in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif')

# endregion

# region # EXTENT AND CLIPPING ---------------------------------------------------------------
# Get extent of subbasin area and create subbasin mask
WS = rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin)+'_watershed_no_lakes_1m.tif')
extent = WS.bounds
bounds = (extent[0], extent[1], extent[2], extent[3])
rows=WS.shape[0]
cols=WS.shape[1]
[R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin)+'_watershed_no_lakes_1m.tif'))
nans=np.where(WS<1)
WS[nans]='nan'

# Clip model input data to subbasin boundaries - set values outside subbasin to NaN so that they aren't included in calculations
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/')
inputs = ['_Slope','_Eastness','_Northness','','_Curvature','_Aspect'] # Input model data. The empty '' is for the bare earth file
for m in range(len(inputs)):
    [R,x]=np.array(pyrsgis.raster.read(str(subbasin)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif'))
    x=x*WS
    pyrsgis.raster.export(x, R, filename='clipped/'+str(subbasin)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output')
for n in range(len(phases)):
    [R,x]=np.array(pyrsgis.raster.read('Distributed_Xt_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif'))
    x=x*WS
    pyrsgis.raster.export(x, R, filename='clipped/Distributed_Xt_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif')

del m

# Clip final snowdepth (from Watershed) to extent of the subbasin area
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Final/resolution_'+str(resolution)+'m/'+str(year))
for n in range(len(phases)):
    fn_in = str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth.tif'
    fn_out = str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth.tif'
    gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS="EPSG:32610 - WGS 84 / UTM zone 10N")
del fn_in,fn_out
gc.collect()

# Clip data to subbasin boundaries - set values outside subbasin to NaN so that they aren't included in calculations
for n in range(len(phases)):
    [R,x]=np.array(pyrsgis.raster.read(str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth.tif'))
    x=x*WS
    pyrsgis.raster.export(x, R, filename='clipped/'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth.tif')
del n

# Clip model input data (from Watershed) to extent of the subbasin area
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/')
inputs = ['_Slope','_Eastness','_Northness','','_Curvature','_Aspect'] # Input model data. The empy '' is for the bare earth file
for m in range(len(inputs)):
    fn_in = str(watershed)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif'
    fn_out = str(subbasin)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif'
    gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS="EPSG:32610 - WGS 84 / UTM zone 10N")
del fn_in,fn_out
gc.collect()
# endregion