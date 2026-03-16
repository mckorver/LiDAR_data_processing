# Downsample, clip, or set extents of rasters
# This script is NOT meant to be run in its entirety, only selected code chunks when needed.

from osgeo import gdal
import os
import subprocess
import numpy as np
import pandas as pd
import pyrsgis
import geopandas

# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/input_data/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
resolution2 = var['resolution2'][0]
BEversion = var['BEversion'][0]
CANversion = var['CANversion'][0]
glaciers = var['glaciers'][0]
glaciermodel = var['glaciermodel'][0]
lakemodel = var['lakemodel'][0]
phases = []
subbasin = []
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if isinstance(y, float):
            a.append(int(y))
        else:
            a.append(y)
append_fun(phases,'phases')
append_fun(subbasin,'subbasin')

# region # DOWNSAMPLING -----------------------------------------------------------
# NOTE continuous data are resampled with averaging, binary data with nearest neighbor. All nodata values are set to -9999
# Downsample snowdepth to 2 m
for n in range(len(phases)):
    if glaciers=='Y':
        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution1)+'m/')
        cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff '+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.tif'
        subprocess.run([x for x in cmd.split(" ") if x != ""])
        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution2)+'m/')
        [R,SD]=np.array(pyrsgis.raster.read(str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.tif', bands='all'))
        pyrsgis.raster.export(SD, R, filename=str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.tif')
    else:
        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution1)+'m/')
        cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff '+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif'
        subprocess.run([x for x in cmd.split(" ") if x != ""])
        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution2)+'m/')
        [R,SD]=np.array(pyrsgis.raster.read(str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif', bands='all'))
        pyrsgis.raster.export(SD, R, filename=str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif')

# Downsample Xt to 2 m
for m in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution1)+'m')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff Distributed_Xt_'+str(extent)+'_'+str(year)+'_'+str(phases[m])+'.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution2)+'m/Distributed_Xt_'+str(extent)+'_'+str(year)+'_'+str(phases[m])+'.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution2)+'m')
    [R,y]=np.array(pyrsgis.raster.read('Distributed_Xt_'+str(extent)+'_'+str(year)+'_'+str(phases[m])+'.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename='Distributed_Xt_'+str(extent)+'_'+str(year)+'_'+str(phases[m])+'.tif')

# Downsample DEM model input data to 2 m
inputs = ['_Slope','_Eastness','_Northness','','_Curvature','_Aspect'] # Input model data. The empy '' is for the bare earth file
for m in range(len(inputs)):
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff '+str(extent)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif '+str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(extent)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(extent)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif')

# Downsample canopy model input data to 2 m
inputs = ['_CC','_CH', '_CD'] # Input model data.
for m in range(len(inputs)):
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r average -dstnodata "-9999" -of GTiff '+str(extent)+str(inputs[m])+'_v'+str(CANversion)+'_'+str(resolution1)+'m.tif '+str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+str(inputs[m])+'_v'+str(CANversion)+'_'+str(resolution2)+'m.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(extent)+str(inputs[m])+'_v'+str(CANversion)+'_'+str(resolution2)+'m.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(extent)+str(inputs[m])+'_v'+str(CANversion)+'_'+str(resolution2)+'m.tif')

# Downsample lakes to 2 m
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/')
cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(extent)+'_lakes_'+str(resolution1)+'m.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution2)+'m/'+str(extent)+'_lakes_'+str(resolution2)+'m.tif'
subprocess.run([x for x in cmd.split(" ") if x != ""])

# Downsample glaciers to 2 m
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/')
cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(extent)+'_glaciers_'+str(resolution1)+'m.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution2)+'m/'+str(extent)+'_glaciers_'+str(resolution2)+'m.tif'
subprocess.run([x for x in cmd.split(" ") if x != ""])

# Downsample Watershed Mask to 2 m
lakes = ''
glaciers = ''
y = 0
for m in range(len(subbasin)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(subbasin[y])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution1)+'m.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution2)+'m/'+str(subbasin[y])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution2)+'m.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(subbasin[y])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution2)+'m.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(subbasin[y])+'_watershed'+str(lakes)+str(glaciers)+'_'+str(resolution2)+'m.tif')

# Downsample SFA to 2 m
for n in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution1)+'m/')
    cmd='gdalwarp -overwrite -tr 2.0 2.0 -r near -dstnodata "-9999" -of GTiff '+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif '+str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif'
    subprocess.run([x for x in cmd.split(" ") if x != ""])
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/')
    [R,y]=np.array(pyrsgis.raster.read(str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif', bands='all'))
    pyrsgis.raster.export(y, R, filename=str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif')
# endregion

# region # EXTENT AND CLIPPING ---------------------------------------------------------------
# set projection and nodata value of output
projection="EPSG:32609 - WGS 84 / UTM zone 9N"
nodata = -9999
phase = 'P3'
filename = '25_3010_03_TsitikaWS_DEM_1m_NGF_WGS84_UTM09_Ellips_FullExtent.tif'

# Get the desired extent. You need a shapefile (polygon) of the area for this.
WS = geopandas.read_file(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/vector/'+str(extent)+'_watershed_outlines.shp')
ext = WS.bounds
bounds = (ext['minx'][0], ext['miny'][0], ext['maxx'][0], ext['maxy'][0])
rows=ext['maxy'][0] - ext['miny'][0]
cols=ext['maxx'][0] - ext['minx'][0]

# Clip LiDAR data to extent of the desired area
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Input_data/LiDAR_data/'+str(watershed)+'/'+str(year)+'/')
fn_in = filename
fn_out = str(extent)+'_'+str(year)+'_'+str(phase)+'_SS.tif'
gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# Clip input data (from Watershed) to extent of the subbasin area
# Watershed masks
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/')
inputs = ['_','_no_lakes_','_no_glaciers_','_no_lakes_no_glaciers_']
for m in range(len(inputs)):
    fn_in = str(watershed)+'_watershed'+str(inputs[m])+str(resolution1)+'m.tif'
    fn_out = str(extent)+'_watershed'+str(inputs[m])+str(resolution1)+'m.tif'
    gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# Bare Earth and BE-derived products
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/')
inputs = ['_Slope','_Eastness','_Northness','','_Curvature','_Aspect'] # The empy '' is for the bare earth file
for m in range(len(inputs)):
    fn_in = str(watershed)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif'
    fn_out = str(extent)+str(inputs[m])+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif'
    gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# Canopy metrics
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/')
inputs = ['_CC','_CD','_CH']
for m in range(len(inputs)):
    fn_in = str(watershed)+str(inputs[m])+'_v'+str(CANversion)+'_'+str(resolution1)+'m.tif'
    fn_out = str(extent)+str(inputs[m])+'_v'+str(CANversion)+'_'+str(resolution1)+'m.tif'
    gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# Lake and glacier masks
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/')
inputs = ['_lakes_','_glaciers_','_lakes_and_glaciers_']
for m in range(len(inputs)):
    fn_in = str(watershed)+str(inputs[m])+str(resolution1)+'m.tif'
    fn_out = str(extent)+str(inputs[m])+str(resolution1)+'m.tif'
    gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# Road masks
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Roads_mask/')
fn_in = str(watershed)+'_ROADS.tif'
fn_out = str(extent)+'_ROADS.tif'
gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# Snow-free elevation masks
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution1)+'m/')
for n in range(len(phases)):
    fn_in = str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif'
    fn_out = str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SFA.tif'
    gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# Meteorological parameters
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution1)+'m/')
inputs = ['Distributed_PDD_','Distributed_Snowfall_','Distributed_Xt_']
for m in range(len(inputs)):
    for n in range(len(phases)):
        fn_in = str(inputs[m])+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'.tif'
        fn_out = str(inputs[m])+str(extent)+'_'+str(year)+'_'+str(phases[n])+'.tif'
        gdal.Warp(fn_out, fn_in, outputBounds=bounds, width=cols, height=rows, dstSRS=projection, dstNodata=nodata)

# endregion