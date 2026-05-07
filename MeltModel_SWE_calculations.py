# This code calculates SWE
# This code outputs:
# Raster map of SWE and summary statistics:
# Mean snowdepth, snowdensity, SWE, total snow volume, total snow water volume
# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

import numpy as np
import pyrsgis
import os
import pandas as pd

# Import input data -----------------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution2 = var['resolution2'][0]
BEversion = var['BEversion'][0]
date = var['date'][0]
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
resolution3 = 10
future_date = "2026-05-04"

# Import bare earth
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/INPUTS')
[R,BE]=np.array(pyrsgis.raster.read(str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution3)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan

# Import SWE and melt data (in m) - clip to extent
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/SWE_EST')
[R,SWE_start]=np.array(pyrsgis.raster.read('SWE_est_from_DOY110_to_DOY123_canopy_DDF2p0.tif', bands=1))
SWE_start=SWE_start*1000
[R,melt_act_start]=np.array(pyrsgis.raster.read('SWE_est_from_DOY110_to_DOY123_canopy_DDF2p0.tif', bands=2))
[R,melt_pot_start]=np.array(pyrsgis.raster.read('SWE_est_from_DOY110_to_DOY123_canopy_DDF2p0.tif', bands=3))

# Calculations ---------------------------------------------------------------------------------------------------------------------
total_SWV=[]
mean_SWE_depth=[]
total_melt_act=[]
total_melt_pot=[]
banded_mean_SWE=[] # mm
banded_total_SWV=[] # convert to m3
elevation_bands=[]
for a in range(len(subbasin)):
    # Clip SnowDepth, SnowDensity, SWE map by subbasin:
    if lakemodel == 'Y' and (glaciermodel == 'Y' or glaciers == 'N'):
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution3)+'m/'+str(subbasin[a])+'_watershed_'+str(resolution3)+'m.tif'))
    elif lakemodel == 'N' and (glaciermodel == 'Y' or glaciers == 'N'):
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution3)+'m/'+str(subbasin[a])+'_watershed_no_lakes_'+str(resolution3)+'m.tif'))
    elif lakemodel == 'N' and glaciermodel == 'N':
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution3)+'m/'+str(subbasin[a])+'_watershed_no_lakes_no_glaciers_'+str(resolution3)+'m.tif'))
    nans=np.where(WS<1)
    WS[nans]=np.nan
    SWE=SWE_start*WS
    melt_act=melt_act_start*WS
    melt_pot=melt_pot_start*WS

    # Calculate total SWV volume (m3) and mean SWE depth (mm)
    x=np.nansum(SWE*int(resolution3)*int(resolution3))/1000
    total_SWV.append(x)
    y=np.nanmean(SWE)
    mean_SWE_depth.append(y)
    del x,y

    # Calculate total actual melt volume
    x=(np.nansum(melt_act)*int(resolution3)*int(resolution3))
    total_melt_act.append(x)
    del x

    # Calculate total potential melt volume
    x=(np.nansum(melt_pot)*int(resolution3)*int(resolution3))
    total_melt_pot.append(x)
    del x

    # Calculate min and max subbasin elevations
    x=BE*WS
    min=np.nanmin(x)
    min_elev=np.floor(min/100).astype(int)*100 #This is the minimum elevation of the bare earth, rounded down to the first hundred
    max=np.nanmax(x)
    max_elev=np.ceil(max/100).astype(int)*100 + 1 #This is the maximum elevation of the bare earth, rounded up to the first hundred and +1
    elev_bands=np.arange(start=min_elev, stop=max_elev, step=100) # Get elev bands

    # Bin SWE data by elevation for each subbasin
    y=np.ndarray.flatten(SWE)
    elev_data=np.ndarray.flatten(BE)
    SWE_within_each_band=[]
    for j in range(len(elev_bands)-1):
        lower=elev_bands[j]
        upper=elev_bands[j+1]
        indices=np.where((elev_data>lower) & (elev_data<upper))
        SWE_inband=y[indices]
        SWE_within_each_band.append(SWE_inband)
    mean_SWE_within_band=[] # mm
    total_SWV_within_band=[] # convert to m3
    for k in range(len(SWE_within_each_band)):
        mean_SWE=np.nanmean(SWE_within_each_band[k])
        mean_SWE_within_band.append(mean_SWE)
        total_SWE=np.nansum(SWE_within_each_band[k]*int(resolution3)*int(resolution3))/1000
        total_SWV_within_band.append(total_SWE)
    banded_mean_SWE.append(mean_SWE_within_band)
    banded_total_SWV.append(total_SWV_within_band)

    # Set new elev bands for table
    min_elev2=min_elev+50
    max_elev2=max_elev-50
    elev_bands=np.arange(start=min_elev2, stop=max_elev2, step=100)
    elevation_bands.append(elev_bands)

# Output ------------------------------------------------------------------------------------------
# Export key numbers 
if glaciers == 'Y':
    path = str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Key_numbers/'
    os.makedirs(path, exist_ok=True) 
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Key_numbers/')
else:
    path = str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Key_numbers/'
    os.makedirs(path, exist_ok=True)
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Key_numbers/')
mean_SWE_depth=pd.DataFrame(list(zip(subbasin,mean_SWE_depth)),columns=['Watershed','Mean_SWE_mm'])
mean_SWE_depth.to_csv('Mean_SWE_'+str(future_date)+'.csv') 
total_SWV=pd.DataFrame(list(zip(subbasin,total_SWV)),columns=['Watershed','Total_SWV_m3'])
total_SWV.to_csv('Total_SWV_'+str(future_date)+'.csv')
total_meltact=pd.DataFrame(list(zip(subbasin,total_melt_act)),columns=['Watershed','Total_melt_act_m3'])
total_meltact.to_csv('Total_meltact_'+str(future_date)+'.csv')
total_meltpot=pd.DataFrame(list(zip(subbasin,total_melt_pot)),columns=['Watershed','Total_melt_pot_m3'])
total_meltpot.to_csv('Total_meltpot_'+str(future_date)+'.csv')

# Export elevation-banded total SWV (m3)
if glaciers == 'Y':
    path = str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Elevation_banded_water_volumes/'
    os.makedirs(path, exist_ok=True) 
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Elevation_banded_water_volumes/')
else:
   path = str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Elevation_banded_water_volumes/'
   os.makedirs(path, exist_ok=True)
   os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Figure_table_production/Future_SWE/OUTPUTS/Elevation_banded_water_volumes/')
for m in range(len(subbasin)):
    x=np.array(banded_mean_SWE[m])
    x=x.reshape(len(x),)
    y=np.array(banded_total_SWV[m])
    y=y.reshape(len(y),)
    z=np.transpose(np.array([elevation_bands[m],x,y]))
    zz=pd.DataFrame(z,columns=['elev_band','SWE_mean_mm','Total_SWV_m3'])
    zz['watershed'] = subbasin[m]
    zz['date'] = future_date
    zz.to_csv(str(subbasin[m])+'_Elevation_banded_'+str(future_date)+'.csv')