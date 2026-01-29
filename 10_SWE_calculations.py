# This code calculates SWE
# This code outputs:
# Raster map of SWE and summary statistics:
# Mean snowdepth, snowdensity, SWE, total snow volume, total snow water volume
# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

# 'CRU','Comox','Eric','Moat','Rees','Residual'
# 'MV','SEY','CAP','BurwellLake','LochLomond','PalisadeLake','UpperSeymour'
# 'ENG','Arrowsmith','Fishtail','Cokely'
# 'TSI', 'RussellCreek'

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin=['CRU','Comox','Eric','Moat','Rees','Residual'] #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2024' # Enter year of interest
phases=['P1'] # Enter survey phases ('P1','P2', etc.)
BEversion = 2 # Enter Bare Earth version number
resolution = 2 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
lakemodel = 'N' # Enter 'Y' or 'N' for including modelled SnowDepth on lakes
glaciermodel = 'N' # Enter 'Y' or 'N' for including a SWE model for glaciers, or 'NA' if the watershed does not have glaciers

import numpy as np
import pyrsgis
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import input data -----------------------------------------------------------------------------
# Import watershed mask
[R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_'+str(resolution)+'m.tif'))
nans=np.where(WS<1)
WS[nans]=np.nan

# Import bare earth
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan

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

Density=[]
Depth=[]
for n in range(len(phases)):
    # Import snow depth data (in m) - clip to study area
    [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'.tif', bands='all'))
    nosnow=np.where(x==0) #identify pixels where snow=0m. We set density also to 0 here
    nans=np.where(x<0)
    x[nans]=0 #set all nans to 0
    bsl=np.where(SFAs[n]==1)
    x[bsl]=0 #set all pixels below snowline to 0
    x=x*WS #only keep pixels with value=1 in watershed mask
    Depth.append(x)
        
    # Import snow density data (in g/cm3) - clip to study area
    # set snow free areas (i.e., nans) to 0 for mean SWE calcs (so that mean SWE is calculated across whole basin, not just snow-covered areas)
    [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'.tif', bands='all'))
    x[nosnow]=0 #set all 0m snow pixels to 0 density
    nans=np.where(x<0)
    x[nans]=0 #set all nans to 0
    bsl=np.where(SFAs[n]==1)
    x[bsl]=0 #set all pixels below snowline to 0
    x=x*WS #only keep pixels with value=1 in watershed mask 
    Density.append(x)
del x,n,bsl

# Produce SWE maps (in mm) ---------------------------------------------------------------------------------------
SWE=[]
for n in range(len(phases)):
    x=((Depth[n]*Density[n]))*1000
    SWE.append(x)
del x,n

if glaciermodel == 'Y':
    [R,WS_no_lakes]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_no_lakes_'+str(resolution)+'m.tif'))
    nans=np.where(WS_no_lakes<1)
    WS_no_lakes[nans]=np.nan
    
    # Import eastness, northness, and slope
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

    # Flatten model input datasets
    BE_flattened=np.ndarray.flatten(BE)
    Eastness_flattened=np.ndarray.flatten(Eastness)
    Northness_flattened=np.ndarray.flatten(Northness)
    Slope_flattened=np.ndarray.flatten(Slope)
    del Eastness,Northness,Slope

    # Run modelling-based gap-filling for each phase and each subbasin
    SWE_filled=[]
    for n in range(len(phases)):
        # Determine areas where interpolation is required (i.e. within the boundary of the watershed, not in lakes) and set these areas to 0 and everything else to 1
        WS_mask_flattened=np.ndarray.flatten(WS_no_lakes)
        SWE_flattened=np.ndarray.flatten(SWE[n])
        modelling_areas=(SWE_flattened/SWE_flattened).astype('float64')
        i=np.ndarray.flatten(np.array(np.where(WS_mask_flattened==1))).astype('float64')
        j=np.ndarray.flatten(np.argwhere(np.isnan(SWE_flattened))).astype('float64')
        k=(np.intersect1d(i,j))
        k=k.astype('int64')
        modelling_areas[k]=100
        l=np.where(modelling_areas==1)
        model_building_SWE=SWE_flattened[l]
        model_building_elevation=BE_flattened[l]
        model_building_eastness=Eastness_flattened[l]
        model_building_northness=Northness_flattened[l]    
        model_building_slope=Slope_flattened[l]
        model_inference_elevation=BE_flattened[k]
        model_inference_eastness=Eastness_flattened[k]
        model_inference_northness=Northness_flattened[k]
        model_inference_slope=Slope_flattened[k]
        del WS_mask_flattened,i,j,l,modelling_areas
                
        # Delete pixels where no input variables are available for model building
        nans=np.argwhere(np.isnan(model_building_eastness))
        model_building_eastness=np.delete(model_building_eastness,nans)
        model_building_northness=np.delete(model_building_northness,nans)
        model_building_slope=np.delete(model_building_slope,nans)
        model_building_elevation=np.delete(model_building_elevation,nans)
        model_building_SWE=np.delete(model_building_SWE,nans)
        nans=np.argwhere(np.isnan(model_building_northness))
        model_building_eastness=np.delete(model_building_eastness,nans)
        model_building_northness=np.delete(model_building_northness,nans)
        model_building_slope=np.delete(model_building_slope,nans)
        model_building_elevation=np.delete(model_building_elevation,nans)
        model_building_SWE=np.delete(model_building_SWE,nans)
        nans=np.argwhere(np.isnan(model_building_elevation))
        model_building_eastness=np.delete(model_building_eastness,nans)
        model_building_northness=np.delete(model_building_northness,nans)
        model_building_slope=np.delete(model_building_slope,nans)
        model_building_elevation=np.delete(model_building_elevation,nans)
        model_building_SWE=np.delete(model_building_SWE,nans)
                
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
        y_variables=model_building_SWE
        x_train,x_test,y_train,y_test=train_test_split(x_variables,y_variables,test_size=0.2, random_state=12345)
        MLR_model=LinearRegression().fit(x_train,y_train)
        MLR_model.fit(x_train,y_train)
        pred_test=MLR_model.predict(x_test)
        rmse=np.sqrt(mean_squared_error(y_test,pred_test))
        del x_variables,y_variables,x_train,x_test,model_building_SWE,model_building_elevation,model_building_eastness,model_building_northness,model_building_slope,pred_test
                
        # Apply MLR model to gaps
        x_variables_inference=np.transpose([model_inference_elevation,model_inference_northness,model_inference_eastness,model_inference_slope])
        modelled_y=MLR_model.predict(x_variables_inference)
        SWE_flattened[k]=modelled_y
        del x_variables_inference,k,model_inference_eastness,model_inference_northness,model_inference_slope,MLR_model
                
        # Cap any anomylously high or low modelled values
        errors=np.where(SWE_flattened<0)
        SWE_flattened[errors]=np.nan
        errors=np.where(SWE_flattened>5000)
        SWE_flattened[errors]=5000    
                
        # Re-enter nans and reshape results
        nans=np.where(model_inference_elevation==-9999)
        SWE_flattened[nans]=np.nan
        dims=np.shape(SWE[n])
        x=np.reshape(SWE_flattened,(dims[0],dims[1]))
        SWE_filled.append(x)
        del dims,SWE_flattened
    SWE=SWE_filled
    del WS_no_lakes
del WS

# Calculations ---------------------------------------------------------------------------------------------------------------------
min_elevs=[] # These are here just for reference: the min elevations of each subbasin
max_elevs=[] # These are here just for reference: the max elevations of each subbasin
for a in range(len(subbasin)):
    SWE_sub=[]
    Depth_sub=[]
    Density_sub=[]
    for n in range(len(phases)):
        # Clip SnowDepth, SnowDensity, SWE map by subbasin:
        if lakemodel == 'Y' and (glaciermodel == 'Y' or glaciermodel == 'NA'):
            [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin[a])+'_watershed_'+str(resolution)+'m.tif'))
        elif lakemodel == 'N' and (glaciermodel == 'Y' or glaciermodel == 'NA'):
            [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin[a])+'_watershed_no_lakes_'+str(resolution)+'m.tif'))
        elif lakemodel == 'N' and glaciermodel == 'N':
            [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin[a])+'_watershed_no_lakes_no_glaciers_'+str(resolution)+'m.tif'))
        nans=np.where(WS<1)
        WS[nans]=np.nan
        x=SWE[n]*WS
        y=Depth[n]*WS
        z=Density[n]*WS
        SWE_sub.append(x)
        Depth_sub.append(y)
        Density_sub.append(z)
    del x,y,z

    # Calculate min and max subbasin elevations
    x=BE*WS
    min=np.nanmin(x)
    min_elev=np.floor(min/100).astype(int)*100 #This is the minimum elevation of the bare earth, rounded down to the first hundred
    max=np.nanmax(x)
    max_elev=np.ceil(max/100).astype(int)*100 + 1 #This is the maximum elevation of the bare earth, rounded up to the first hundred and +1

    # Calculate mean snow depth
    mean_Depth=[]
    for n in range(len(phases)):
        x=np.nanmean(Depth_sub[n])
        mean_Depth.append(x)
    del n,x

    # Calculate mean snow depth above snowline
    mean_Depth_above=[]
    for n in range(len(phases)):
        x=Depth_sub[n]
        y=SFAs[n]
        x[y==1]=np.nan #set all pixels below snowline to NaN
        z=np.nanmean(x)
        Depth_sub[n]=x
        mean_Depth_above.append(z)
    del n,x
                
    # Calculate total SWV volume (m3) and mean SWE depth (mm)
    total_SWV=[]
    mean_SWE_depth=[]
    for n in range(len(phases)):
        x=np.nansum(SWE_sub[n]*resolution*resolution)/1000
        total_SWV.append(x)
        y=np.nanmean(SWE_sub[n])
        mean_SWE_depth.append(y)
    del n,x,y

    # Calculate total snow volume, clip out areas below snowline
    total_SV=[]
    for n in range(len(phases)):
        x=Depth_sub[n]
        z=(np.nansum(Depth_sub[n])*resolution*resolution)
        total_SV.append(z)
    del n
            
    # Set snow free areas to NaN for mean density calcs (so that density is only calculated across snow-covered areas)
    for n in range(len(phases)):
        x=Density_sub[n]
        zeros=np.where(x==0)
        x[zeros]=np.nan #set all 0 to nans
        Density_sub[n]=x
    del n,x

    # Calculate mean snow density (convert to kg/m3)
    mean_Density=[]
    for n in range(len(phases)):
        x=np.nanmean(Density_sub[n])*1000
        mean_Density.append(x)
    del n,x

    # Get elev bands
    elev_bands=np.arange(start=min_elev, stop=max_elev, step=100)

    # Bin SWE data by elevation for each subbasin
    banded_mean_SWE=[] # mm
    banded_total_SWV=[] # convert to m3
    for m in range(len(SWE_sub)):
        y=np.ndarray.flatten(SWE_sub[m])
        elev_data=np.ndarray.flatten(BE)
        SWE_within_each_band=[]
        for j in range(len(elev_bands)-1):
            lower=elev_bands[j]
            upper=elev_bands[j+1]
            indices=np.where((elev_data>lower) & (elev_data<upper))
            SWE_inband=y[indices]
            SWE_within_each_band.append(SWE_inband)
        mean_SWE_banded=[]
        total_SWV_banded=[]
        for k in range(len(SWE_within_each_band)):
            mean_SWE=np.nanmean(SWE_within_each_band[k])
            mean_SWE_banded.append(mean_SWE)
            total_SWE=np.nansum(SWE_within_each_band[k]*resolution*resolution)/1000
            total_SWV_banded.append(total_SWE) 
        banded_mean_SWE.append(mean_SWE_banded)
        banded_total_SWV.append(total_SWV_banded)

    # Set new elev bands for table
    min_elev2=min_elev+50
    max_elev2=max_elev-50
    elev_bands=np.arange(start=min_elev2, stop=max_elev2, step=100)

    # OPTIONAL set mean density of P1 to 332.4
    #mean_Density[0]=332.4

    # Output ------------------------------------------------------------------------------------------
    # Export SWE maps
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SWE/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m')
    if subbasin[a]==watershed:
        for n in range(len(phases)):
            pyrsgis.raster.export(SWE_sub[n], R, filename=str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SWE_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.tif')
    #if subbasin[a]!=watershed:
    #    for n in range(len(phases)):
    #        pyrsgis.raster.export(SWE[n], R, filename=str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_SWE.tif')
  
    # Export key numbers 
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Key_numbers/'+str(year)+'/resolution_'+str(resolution)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/')
    mean_Depth=pd.DataFrame(list(zip(phases,mean_Depth)),columns=['Survey','Mean_snow_depth_m'])
    mean_Depth.to_csv(str(subbasin[a])+'_'+str(year)+'_Mean_snow_depth.csv')
    mean_Depth_above=pd.DataFrame(list(zip(phases,mean_Depth_above)),columns=['Survey','Mean_snow_depth_aboveSL_m'])
    mean_Depth_above.to_csv(str(subbasin[a])+'_'+str(year)+'_Mean_snow_depth_aboveSL.csv')
    mean_Density=pd.DataFrame(list(zip(phases,mean_Density)),columns=['Survey','Mean_snow_density_kgm3'])
    mean_Density.to_csv(str(subbasin[a])+'_'+str(year)+'_Mean_snow_density.csv')
    mean_SWE_depth=pd.DataFrame(list(zip(phases,mean_SWE_depth)),columns=['Survey','Mean_SWE_mm'])
    mean_SWE_depth.to_csv(str(subbasin[a])+'_'+str(year)+'_Mean_SWE.csv') 
    total_SV=pd.DataFrame(list(zip(phases,total_SV)),columns=['Survey','Total_snow_m3'])
    total_SV.to_csv(str(subbasin[a])+'_'+str(year)+'_Total_snow_volume.csv')
    total_SWV=pd.DataFrame(list(zip(phases,total_SWV)),columns=['Survey','Total_SWV_m3'])
    total_SWV.to_csv(str(subbasin[a])+'_'+str(year)+'_Total_SWV.csv')

    # Export elevation-banded total SWV (m3)
    for m in range(len(banded_total_SWV)):
        y=np.array(banded_total_SWV[m])
        y=y.reshape(len(y),)
        z=np.transpose(np.array([elev_bands,y]))
        zz=pd.DataFrame(z,columns=['elev_band','Total_SWV_m3'])
        zz.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[m])+'_Elevation_banded_total_SWV.csv')
            
    # Save elevation-banded mean SWE
    for m in range(len(banded_mean_SWE)):
        y=np.array(banded_mean_SWE[m])
        y=y.reshape(len(y),)
        z=np.transpose(np.array([elev_bands,y]))
        zz=pd.DataFrame(z,columns=['elev_band','SWE_mean_mm'])
        zz.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[m])+'_Elevation_banded_mean_SWE.csv')
    min_elevs.append(min)
    max_elevs.append(max)