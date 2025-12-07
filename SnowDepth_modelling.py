# This code fills in spatial data gaps with modelled Snow Depth using a random forest model

# 'CRU','Comox','Eric','Moat','Rees','Residual'
# 'MV','SEY','CAP','BurwellLake','LochLomond','PalisadeLake','UpperSeymour'
# 'ENG','Arrowsmith','Fishtail','Cokely'
# 'TSI', 'RussellCreek'

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='MV' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='CAP' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2024' # Enter year of interest
phases=['P1','P2','P3'] # Enter survey phases ('P1','P2', etc.)
BEversion = 6 # Enter Bare Earth version number
resolution = 1 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
lakemodel = 'N' # Enter 'Y' or 'N' for including modelled SnowDepth on lakes
glaciermodel = 'NA' # Enter 'Y' or 'N' for including a SWE model for glaciers, or 'NA' if the watershed does not have glaciers

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
# Import watershed masks
[R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_'+str(resolution)+'m.tif'))
nans=np.where(WS<1)
WS[nans]=np.nan

[R,WS_no_lakes]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_no_lakes_'+str(resolution)+'m.tif'))
nans=np.where(WS_no_lakes<1)
WS_no_lakes[nans]=np.nan

# Import masks that delineate the areas for gapfilling    
gapfill = []
for n in range(len(phases)):
    [R,gap_area]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Manual_corrections/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_gapfilling_area.tif'))
    nans=np.where(gap_area==0)
    gap_area[nans]=np.nan
    gapfill.append(gap_area)

# Import elevation, eastness, northness, and slope
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan
BE=BE*WS_no_lakes # Remove input data for lake areas 
[R,Eastness]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
nans=np.where(Eastness<-100)
Eastness[nans]=np.nan
Eastness=Eastness*WS_no_lakes # Remove input data for lake areas
[R,Northness]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
nans=np.where(Northness<-100)
Northness[nans]=np.nan
Northness=Northness*WS_no_lakes # Remove input data for lake areas
[R,Slope]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
nans=np.where(Slope<-100)
Slope[nans]=np.nan
Slope=Slope*WS_no_lakes # Remove input data for lake areas
del nans

# Import snow free areas. A raster mask is used if available, otherwise an elevation threshold.
SFAs=[]
for n in phases:
    file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(n)+'_SFA.tif')
    if file.is_file():
        [R,SFA]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(n)+'_SFA.tif', bands='all'))
    else:
        x=pd.read_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/SFETs_'+str(watershed)+'_'+str(year)+'.csv')
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
    if lakemodel == 'Y':
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodelY.tif', bands='all'))
    elif lakemodel == 'N':
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodelN.tif', bands='all'))
    nans=np.where(x<0)
    x[nans]=np.nan #set all nans
    x=x*WS #only keep pixels with value=1 in watershed mask
    Depth.append(x)
del x,n

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
    # Determine areas where interpolation is required (i.e. within the gapfill area) and set these areas to 0 and everything else to 1
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
    del gapfill_flattened,i,j,l,modelling_areas
                    
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
                
    # Re-enter nans and reshape results
    nans=np.where(model_inference_elevation==-9999)
    Depth_flattened[nans]=np.nan
    dims=np.shape(Depth[n])
    x=np.reshape(Depth_flattened,(dims[0],dims[1]))
    bsl=np.where(SFAs[n]==1)
    x[bsl]=0 #set all pixels below snowline to 0
    x=x*WS
    
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
        [R,lakemask]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_lakes_'+str(resolution)+'m.tif', bands='all'))
        i=np.where(lakemask==1)
        filled[i]= np.nan

    Depth_filled.append(filled)
    del dims,Depth_flattened

# Output ------------------------------------------------------------------------------------------
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m')
for n in range(len(phases)):
    pyrsgis.raster.export(Depth_filled[n], R, filename=str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_filled.tif')
