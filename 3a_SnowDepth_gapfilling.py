# This code fills small gaps with linear interpolation and can model lake surface area snow cover based on snowcover of surrounding land
# Outputs:
# Gap-filled snowdepth maps for each phase

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='MV' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='MV' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2024' # Enter year of interest
phases=['P1','P2','P3'] # Enter survey phases ('P1','P2', etc.)
BEversion = 6 # Enter Bare Earth version number.
resolution = 1 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
veg_correction='vegcorrected' # Enter 'vegcorrected' if you want to use the vegetation corected version and '' if not.
lakemodel = 'N' # Enter 'Y' or 'N' for including modelled SnowDepth on lakes
glaciers = 'N' # Enter 'Y' if the watershed has glaciers, 'N' if not
frac = 0.3 # The fraction of assumed lake snowdepth compared to surrounding land

# Import packages
import pyrsgis
import numpy as np
import os
import geopandas
import rasterio
import rasterio.fill
from rasterio import features
from rasterstats import zonal_stats

# Import input data -----------------------------------------------------------
# Import bare earth
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan
del nans

# Import watershed (or subbasin) mask without lakes and glaciers.
if glaciers == 'Y':
    [R,WS_no_lakes]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin)+'_watershed_no_lakes_no_glaciers_'+str(resolution)+'m.tif'))
else:
    [R,WS_no_lakes]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin)+'_watershed_no_lakes_'+str(resolution)+'m.tif'))
nans=np.where(WS_no_lakes<=0)
WS_no_lakes[nans]=np.nan

# Import watershed mask
[R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_'+str(resolution)+'m.tif'))
nans=np.where(WS<1)
WS[nans]=np.nan

# Read lakes vector dataset and create 100m buffer
lakes = geopandas.read_file(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/vector/'+str(watershed)+'_lakes/')
lakes['buffered'] = lakes.buffer(distance=100)

for n in range(len(phases)):
    phase=phases[n]
    # Load Snowdepth raster
    SD_in = rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m/Provisional_SD_'+str(watershed)+'_'+str(year)+'_'+str(phase)+'_capped_clipped_'+str(veg_correction)+'_'+str(resolution)+'m.tif')

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
            str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m/lakes_temp.tif', "w",
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
    [R,land]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m/Provisional_SD_'+str(watershed)+'_'+str(year)+'_'+str(phase)+'_capped_clipped_'+str(veg_correction)+'_'+str(resolution)+'m.tif'))
    [S,lake]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m/lakes_temp.tif'))

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

    # Determine areas where interpolation is required (i.e. within the boundary of the watershed, not in lakes or glaciers) and set these areas to 0 and everything else to 1
    area_mask_flattened=np.ndarray.flatten(WS_no_lakes)
    SD_flattened=np.ndarray.flatten(SD_out)
    interpolation_areas=(SD_flattened/SD_flattened).astype('float64')
    a=np.ndarray.flatten(np.array(np.where(area_mask_flattened==1))).astype('float64')
    b=np.ndarray.flatten(np.argwhere(np.isnan(SD_flattened))).astype('float64')
    k=(np.intersect1d(a,b))
    k=k.astype('int64')
    interpolation_areas[k]=0
    nans=np.argwhere(np.isnan(interpolation_areas))
    interpolation_areas[nans]=1
    dims=np.shape(WS_no_lakes)
    interpolation_areas_2d=np.reshape(interpolation_areas,(dims[0],dims[1]))
    del a,b,k,nans,dims,SD_flattened,area_mask_flattened,interpolation_areas
    
    filled=rasterio.fill.fillnodata(SD_out,interpolation_areas_2d) #enter 'interpolation_areas_2d' if using mask

    if lakemodel == 'N':
        # Remove lakes
        [R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_lakes_'+str(resolution)+'m.tif', bands='all'))
        lakemask[lakemask==0]= np.nan
        i=np.where(lakemask==1)
        filled[i]= np.nan

    # Cut map along watershed boundaries
    filled = filled*WS

    # Output ---------------------------------------------------------------------------------------
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m')
    pyrsgis.export(filled,R,filename='Provisional_SD_'+str(watershed)+'_'+str(year)+'_'+str(phase)+'_capped_clipped'+'_'+str(veg_correction)+'_filled_lakemodel'+str(lakemodel)+'_'+str(resolution)+'m.tif')
    del filled
    
    print('Phase '+str(n+1)+'/'+str(len(phases))+' complete')