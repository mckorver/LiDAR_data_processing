# This code fills in large spatial data gaps with modelled Snow Depth based on elevation, aspect, and slope (random forest model)

import numpy as np
import pyrsgis
import os
import pandas as pd
from pathlib import Path
import rasterio
import rasterio.fill
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import input data -----------------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/input_data/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
date = var['date'][0]
BEversion = var['BEversion'][0]
glaciers = var['glaciers'][0]
glaciermodel = var['glaciermodel'][0]
lakemodel = var['lakemodel'][0]
phases = []
x = var['phases'][var['phases'].notna()]
for n in range(len(x)):
    a = x[n]
    phases.append(a)

# Import watershed mask without lakes
if glaciers == 'Y':
    [R,WS_no_lakes]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_no_lakes_no_glaciers_'+str(resolution1)+'m.tif'))
else:
    [R,WS_no_lakes]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_no_lakes_'+str(resolution1)+'m.tif'))
nans=np.where(WS_no_lakes<=0)
WS_no_lakes[nans]=np.nan

# Import watershed mask
[R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_watershed_'+str(resolution1)+'m.tif'))
nans=np.where(WS<1)
WS[nans]=np.nan

# Import masks that delineate the areas for gapfilling    
gapfill = []
for n in range(len(phases)):
    [R,gap_area]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Manual_corrections/vector/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_missing_areas.tif'))
    nans=np.where(gap_area==0)
    gap_area[nans]=np.nan
    gapfill.append(gap_area)

# Import elevation, eastness, northness, and slope
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan
BE=BE*WS_no_lakes # Remove input data for lake areas 
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

# Import snow free areas. A raster mask is used if available, otherwise an elevation threshold.
SFAs=[]
for n in phases:
    file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(n)+'_SFA.tif')
    if file.is_file():
        [R,SFA]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(n)+'_SFA.tif', bands='all'))
    else:
        x=pd.read_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/SFETs_'+str(extent)+'_'+str(year)+'.csv')
        SFET=x[x['Survey']==n]
        SFET=SFET['SFETs'].iloc[0]
        SFA=BE/BE
        i=np.where(BE<SFET)
        j=np.where(BE>=SFET)
        SFA[i]=1
        SFA[j]=0
    SFAs.append(SFA)

Depth=[]
for n in range(len(phases)):
    # Import snow depth data (in m) - clip to study area
    if glaciers == 'Y':
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_vegcorrected_filled_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'_'+str(resolution1)+'m.tif', bands='all'))
    else:
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_vegcorrected_filled_lakemodel'+str(lakemodel)+'_'+str(resolution1)+'m.tif', bands='all'))
    nans1=np.where(x<0)
    x[nans1]=np.nan
    nans2=np.where(gapfill[n]>0) #remove any pixels that were added through linear interpolation along the border of the gapfill area
    x[nans2]=np.nan
    x=x*WS #only keep pixels with value=1 in watershed mask
    Depth.append(x)
del x,n

# OPTIONAL: if filling the area with a single value (e.g., 0), run this block and skip all further calculations (but save the raster at the end)
#Depth_filled=[]
#for n in range(len(phases)):
#    x=Depth[n]
#    fill=np.where(gapfill[n]>0) #remove any pixels that were added through linear interpolation along the border of the gapfill area
#    x[fill]=0
#    if lakemodel == 'N':
#        [R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_lakes_'+str(resolution1)+'m.tif', bands='all'))
#        i=np.where(lakemask==1)
#        x[i]= np.nan
#    Depth_filled.append(x)

# Calculations ---------------------------------------------
# Flatten model input datasets
BE_flattened=np.ndarray.flatten(BE)
Eastness_flattened=np.ndarray.flatten(Eastness)
Northness_flattened=np.ndarray.flatten(Northness)
Slope_flattened=np.ndarray.flatten(Slope)
del Eastness,Northness,Slope

# Run modelling-based gap-filling for each phase
Depth_filled=[]
for n in range(len(phases)):
    # Determine areas where modelling is required (i.e. within the gapfill area) and set these areas to 0 and everything else to 1
    gapfill_flattened=np.ndarray.flatten(gapfill[n])
    Depth_flattened=np.ndarray.flatten(Depth[n])
    modelling_areas=(Depth_flattened/Depth_flattened).astype('float64')
    i=np.ndarray.flatten(np.array(np.where(gapfill_flattened==1))).astype('float64')
    j=np.ndarray.flatten(np.argwhere(np.isnan(Depth_flattened))).astype('float64')
    k=(np.intersect1d(i,j))
    k=k.astype('int64')
    modelling_areas[k]=100
    l=np.where(modelling_areas==1)
    model_building_Depth=Depth_flattened[l]
    model_building_elevation=BE_flattened[l]
    model_building_eastness=Eastness_flattened[l]
    model_building_northness=Northness_flattened[l]    
    model_building_slope=Slope_flattened[l]
    model_inference_elevation=BE_flattened[k]
    model_inference_eastness=Eastness_flattened[k]
    model_inference_northness=Northness_flattened[k]
    model_inference_slope=Slope_flattened[k]
    del gapfill_flattened,j,l,modelling_areas
                    
    # Delete pixels where no input variables are available for model building
    nans=np.argwhere(np.isnan(model_building_eastness))
    model_building_eastness=np.delete(model_building_eastness,nans)
    model_building_northness=np.delete(model_building_northness,nans)
    model_building_slope=np.delete(model_building_slope,nans)
    model_building_elevation=np.delete(model_building_elevation,nans)
    model_building_Depth=np.delete(model_building_Depth,nans)
    nans=np.argwhere(np.isnan(model_building_northness))
    model_building_eastness=np.delete(model_building_eastness,nans)
    model_building_northness=np.delete(model_building_northness,nans)
    model_building_slope=np.delete(model_building_slope,nans)
    model_building_elevation=np.delete(model_building_elevation,nans)
    model_building_Depth=np.delete(model_building_Depth,nans)
    nans=np.argwhere(np.isnan(model_building_elevation))
    model_building_eastness=np.delete(model_building_eastness,nans)
    model_building_northness=np.delete(model_building_northness,nans)
    model_building_slope=np.delete(model_building_slope,nans)
    model_building_elevation=np.delete(model_building_elevation,nans)
    model_building_Depth=np.delete(model_building_Depth,nans)
                    
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
                    
    # Create MLR model between elevation, northness, eastness, slope and SWE 
    x_variables=np.transpose([model_building_elevation,model_building_northness,model_building_eastness,model_building_slope])
    y_variables=model_building_Depth
    x_train,x_test,y_train,y_test=train_test_split(x_variables,y_variables,test_size=0.2, random_state=12345)
    MLR_model=LinearRegression().fit(x_train,y_train)
    MLR_model.fit(x_train,y_train)
    pred_test=MLR_model.predict(x_test)
    rmse=np.sqrt(mean_squared_error(y_test,pred_test))
    del x_variables,y_variables,x_train,x_test,model_building_Depth,model_building_elevation,model_building_eastness,model_building_northness,model_building_slope,pred_test
                    
    # Apply MLR model to gaps
    x_variables_inference=np.transpose([model_inference_elevation,model_inference_northness,model_inference_eastness,model_inference_slope])
    modelled_y=MLR_model.predict(x_variables_inference)
    Depth_flattened[k]=modelled_y
    del x_variables_inference,k,model_inference_eastness,model_inference_northness,model_inference_slope,MLR_model
                
    # Cap any anomylously high or low modelled values
    errors=np.where(Depth_flattened<0)
    Depth_flattened[errors]=np.nan
    errors=np.where(Depth_flattened>12)
    Depth_flattened[errors]=12  
                
    # Re-enter nans, set below snowline modelling areas to 0, and reshape results
    nans=np.where(model_inference_elevation==-9999)
    Depth_flattened[nans]=np.nan
    SFA_flattened = np.ndarray.flatten(SFAs[n])
    j=np.ndarray.flatten(np.array(np.where(SFA_flattened==1))).astype('float64')
    k=(np.intersect1d(i,j))
    k=k.astype('int64')
    Depth_flattened[k]=0 #set all pixels below snowline in modelling area to 0
    dims=np.shape(Depth[n])
    x=np.reshape(Depth_flattened,(dims[0],dims[1]))
    #x=x*WS
    
    # Determine areas where interpolation is required (i.e. within the boundary of the watershed, not in lakes or glaciers) and set these areas to 0 and everything else to 1
    area_mask_flattened=np.ndarray.flatten(WS)
    SD_flattened=np.ndarray.flatten(x)
    interpolation_areas=(SD_flattened/SD_flattened).astype('float64')
    a=np.ndarray.flatten(np.array(np.where(area_mask_flattened==1))).astype('float64')
    b=np.ndarray.flatten(np.argwhere(np.isnan(SD_flattened))).astype('float64')
    k=(np.intersect1d(a,b))
    k=k.astype('int64')
    interpolation_areas[k]=0
    nans=np.argwhere(np.isnan(interpolation_areas))
    interpolation_areas[nans]=1
    dims=np.shape(WS)
    interpolation_areas_2d=np.reshape(interpolation_areas,(dims[0],dims[1]))
    
    filled=rasterio.fill.fillnodata(x,interpolation_areas_2d) #enter 'interpolation_areas_2d' if using mask

    if lakemodel == 'N':
        [R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_lakes_'+str(resolution1)+'m.tif', bands='all'))
        i=np.where(lakemask==1)
        filled[i]= np.nan

    Depth_filled.append(filled)
    del dims,Depth_flattened

# Output ------------------------------------------------------------------------------------------
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m')
for n in range(len(phases)):
    if glaciers == 'Y':
        pyrsgis.raster.export(Depth_filled[n], R, filename='Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_vegcorrected_filled_modelled_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'_'+str(resolution1)+'m.tif')
    else:
        pyrsgis.raster.export(Depth_filled[n], R, filename='Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_vegcorrected_filled_modelled_lakemodel'+str(lakemodel)+'_'+str(resolution1)+'m.tif')

# Save processing variables
var.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Metadata/'+str(extent)+'_'+str(year)+'_processing_variables_'+str(date)+'.csv', index = False)
