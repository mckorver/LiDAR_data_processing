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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Import input data -----------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/input_data/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
BEversion = var['BEversion'][0]
glaciers = var['glaciers'][0]
glaciermodel = var['glaciermodel'][0]
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

# Import extent mask without lakes (and without glaciers, if applicable).
if glaciers == 'Y':
    [R,WS_gaps]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_no_lakes_no_glaciers_'+str(resolution1)+'m.tif'))
else:
    [R,WS_gaps]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_no_lakes_'+str(resolution1)+'m.tif'))
nans=np.where(WS_gaps<=0)
WS_gaps[nans]=np.nan

# Import extent mask
[R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_'+str(resolution1)+'m.tif'))
nans=np.where(WS<=0)
WS[nans]=np.nan

# Read lakes vector dataset and create 100m buffer
lakes = geopandas.read_file(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/vector/'+str(extent)+'_lakes/')
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

    if glaciermodel == 'Y':
        [R,WS_no_lakes]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_no_lakes_'+str(resolution1)+'m.tif'))
        nans=np.where(WS_no_lakes<1)
        WS_no_lakes[nans]=np.nan

        [R,glaciermask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_glaciers_'+str(resolution1)+'m.tif'))
        nans=np.where(glaciermask<1)
        glaciermask[nans]=np.nan
        
        # Import eastness, northness, and slope
        [R,Eastness]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))
        nans=np.where(Eastness<-100)
        Eastness[nans]=np.nan
        Eastness=Eastness*WS_no_lakes # Remove input data for lake areas
        [R,Northness]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))
        nans=np.where(Northness<-100)
        Northness[nans]=np.nan
        Northness=Northness*WS_no_lakes # Remove input data for lake areas
        [R,Slope]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))
        nans=np.where(Slope<-100)
        Slope[nans]=np.nan
        Slope=Slope*WS_no_lakes # Remove input data for lake areas
        del nans

        # Flatten model input datasets
        Snow_flattened=np.ndarray.flatten(SD_out)
        BE_flattened=np.ndarray.flatten(BE)
        Eastness_flattened=np.ndarray.flatten(Eastness)
        Northness_flattened=np.ndarray.flatten(Northness)
        Slope_flattened=np.ndarray.flatten(Slope)
        del Eastness,Northness,Slope

        # Select pixels to use for model building (i.e. not in glaciers or lakes) and for model inference (i.e., in glaciers)
        WS_mask_flattened=np.ndarray.flatten(WS_gaps)
        available_areas=(Snow_flattened/Snow_flattened).astype('float64')
        a=np.ndarray.flatten(np.array(np.where(WS_mask_flattened==1))).astype('float64')
        b=np.ndarray.flatten(np.argwhere(np.isnan(Snow_flattened))).astype('float64')
        c=(np.intersect1d(a,b))
        c=c.astype('int64')
        available_areas[c]=np.nan
        l=np.where(available_areas==1)
        glacier_flattened=np.ndarray.flatten(glaciermask)
        k=np.where(glacier_flattened==1)
        model_building_Snow=Snow_flattened[l]
        model_building_elevation=BE_flattened[l]
        model_building_eastness=Eastness_flattened[l]
        model_building_northness=Northness_flattened[l]    
        model_building_slope=Slope_flattened[l]
        model_inference_elevation=BE_flattened[k]
        model_inference_eastness=Eastness_flattened[k]
        model_inference_northness=Northness_flattened[k]
        model_inference_slope=Slope_flattened[k]
        del WS_mask_flattened,l,a,b,c
                        
        # Delete pixels where no input variables are available for model building
        nans=np.argwhere(np.isnan(model_building_eastness))
        model_building_eastness=np.delete(model_building_eastness,nans)
        model_building_northness=np.delete(model_building_northness,nans)
        model_building_slope=np.delete(model_building_slope,nans)
        model_building_elevation=np.delete(model_building_elevation,nans)
        model_building_Snow=np.delete(model_building_Snow,nans)
        nans=np.argwhere(np.isnan(model_building_northness))
        model_building_eastness=np.delete(model_building_eastness,nans)
        model_building_northness=np.delete(model_building_northness,nans)
        model_building_slope=np.delete(model_building_slope,nans)
        model_building_elevation=np.delete(model_building_elevation,nans)
        model_building_Snow=np.delete(model_building_Snow,nans)
        nans=np.argwhere(np.isnan(model_building_elevation))
        model_building_eastness=np.delete(model_building_eastness,nans)
        model_building_northness=np.delete(model_building_northness,nans)
        model_building_slope=np.delete(model_building_slope,nans)
        model_building_elevation=np.delete(model_building_elevation,nans)
        model_building_Snow=np.delete(model_building_Snow,nans)
                        
        # Set pixels where input variables are not available for model inference to -9999
        nans=np.argwhere(np.isnan(model_inference_eastness))
        model_inference_eastness[nans]=-9999
        model_inference_northness[nans]=-9999
        model_inference_slope[nans]=-9999
        model_inference_elevation[nans]=-9999
        nan_count=len(nans)
        nans=np.argwhere(np.isnan(model_inference_northness))
        model_inference_eastness[nans]=-9999
        model_inference_northness[nans]=-9999
        model_inference_slope[nans]=-9999
        model_inference_elevation[nans]=-9999
        nan_count=nan_count+len(nans)
        nans=np.argwhere(np.isnan(model_inference_elevation))
        model_inference_eastness[nans]=-9999
        model_inference_northness[nans]=-9999
        model_inference_slope[nans]=-9999
        model_inference_elevation[nans]=-9999
        nan_count=nan_count+len(nans)
                    
        # Create MLR model between elevation, northness, eastness, slope and Snow depth 
        x_variables=np.transpose([model_building_elevation,model_building_northness,model_building_eastness,model_building_slope])
        y_variables=model_building_Snow
        x_train,x_test,y_train,y_test=train_test_split(x_variables,y_variables,test_size=0.2, random_state=12345)
        MLR_model=LinearRegression().fit(x_train,y_train)
        MLR_model.fit(x_train,y_train)
        pred_test=MLR_model.predict(x_test)
        rmse=np.sqrt(mean_squared_error(y_test,pred_test))
        del x_variables,y_variables,x_train,x_test,model_building_Snow,model_building_elevation,model_building_eastness,model_building_northness,model_building_slope,pred_test
                        
        # Apply MLR model to gaps (i.e., glaciers)
        x_variables_inference=np.transpose([model_inference_elevation,model_inference_northness,model_inference_eastness,model_inference_slope])
        modelled_y=MLR_model.predict(x_variables_inference)
        Snow_flattened[k]=modelled_y
        del x_variables_inference,k,model_inference_eastness,model_inference_northness,model_inference_slope,MLR_model
                        
        # Cap any anomylously high or low modelled values
        errors=np.where(Snow_flattened<0)
        Snow_flattened[errors]=np.nan
        errors=np.where(Snow_flattened>12)
        Snow_flattened[errors]=12   
                        
        # Re-enter nans and reshape results
        nans=np.where(model_inference_elevation==-9999)
        Snow_flattened[nans]=np.nan
        dims=np.shape(SD_out)
        x=np.reshape(Snow_flattened,(dims[0],dims[1]))
        SD_out=x
        del dims,Snow_flattened

    # Optional: Import manual corrections masks for noisy areas that were manually outlined (in GIS software) after running the '2_SnowDepth_corrections.py' script
    file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Manual_corrections/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(phase)+'_manual_corrections.tif')
    if file.is_file():
        [R,mask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Manual_corrections/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(phase)+'_manual_corrections.tif', bands='all'))
        mask[mask==0]= np.nan
        i=np.where(mask==1)
        SD_out[i]= np.nan

    # Determine areas where interpolation is required (i.e. within the boundary of the watershed, not in lakes or glaciers) and set these areas to 0 and everything else to 1
    area_mask_flattened=np.ndarray.flatten(WS_gaps)
    SD_flattened=np.ndarray.flatten(SD_out)
    interpolation_areas=(SD_flattened/SD_flattened).astype('float64')
    a=np.ndarray.flatten(np.array(np.where(area_mask_flattened==1))).astype('float64')
    b=np.ndarray.flatten(np.argwhere(np.isnan(SD_flattened))).astype('float64')
    k=(np.intersect1d(a,b))
    k=k.astype('int64')
    interpolation_areas[k]=0
    nans=np.argwhere(np.isnan(interpolation_areas))
    interpolation_areas[nans]=1
    dims=np.shape(WS_gaps)
    interpolation_areas_2d=np.reshape(interpolation_areas,(dims[0],dims[1]))
    del a,b,k,nans,dims,SD_flattened,area_mask_flattened,interpolation_areas
        
    filled=rasterio.fill.fillnodata(SD_out,interpolation_areas_2d) #enter 'interpolation_areas_2d' if using mask

    if lakemodel == 'N':
        # Remove lakes
        [R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_lakes_'+str(resolution1)+'m.tif', bands='all'))
        lakemask[lakemask<=0]= np.nan
        i=np.where(lakemask==1)
        filled[i]= np.nan

    # Cut map along watershed boundaries
    filled = filled*WS

    # Output ---------------------------------------------------------------------------------------
    if glaciers == 'Y':
        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m')
        pyrsgis.export(filled,R,filename='Provisional_SD_'+str(watershed)+'_'+str(year)+'_'+str(phase)+'_capped_clipped_vegcorrected_filled_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'_'+str(resolution1)+'m.tif') 
    else:
        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m')
        pyrsgis.export(filled,R,filename='Provisional_SD_'+str(watershed)+'_'+str(year)+'_'+str(phase)+'_capped_clipped_vegcorrected_filled_lakemodel'+str(lakemodel)+'_'+str(resolution1)+'m.tif') 
    del filled    
    print('Phase '+str(n+1)+'/'+str(len(phases))+' complete')

# Save processing variables
var.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Metadata/'+str(extent)+'_'+str(year)+'_processing_variables.csv')
