# This code calculates SWE
# This code outputs:
# Raster map of SWE and summary statistics:
# Mean snowdepth, snowdensity, SWE, total snow volume, total snow water volume
# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

import numpy as np
import pyrsgis
import os
import pandas as pd
from pathlib import Path

# Import input data -----------------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('V:/LiDAR_data_processing/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution2 = var['resolution2'][0]
BEversion = var['BEversion'][0]
date = var['date'][0]
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

# Import bare earth
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution2)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan

# Import snow free areas. A raster mask is used if available, otherwise an elevation threshold.
SFAs=[]
for n in phases:
    file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(n)+'_SFA.tif')
    if file.is_file():
        [R,SFA]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(n)+'_SFA.tif', bands='all'))
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

# OPTIONAL Read in SWE maps if you want to use a previously created version
#SWE=[]
#for n in range(len(phases)):
#    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SWE/resolution_'+str(resolution2)+'m/Final/')
#    [R,x]=np.array(pyrsgis.raster.read(str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SWE_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif', bands='all')) 
#    nans=np.where(x<0)
#    x[nans]=np.nan
#    x=x*1000
#   SWE.append(x)

Density=[]
Depth=[]
for n in range(len(phases)):
    # Import snow depth data (in m) - clip to extent
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution2)+'m/')
    [R,x]=np.array(pyrsgis.raster.read(str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif', bands='all'))
    nosnow=np.where(x==0) #identify pixels where snow=0m. We set density also to 0 here
    nans=np.where(x<0)
    x[nans]=0 #set all nans to 0
    bsl=np.where(SFAs[n]==1)
    x[bsl]=0 #set all pixels below snowline to 0
    Depth.append(x)
        
    # Import snow density data (in g/cm3) - clip to study area
    # set snow free areas (i.e., nans) to 0 for mean SWE calcs (so that mean SWE is calculated across whole basin, not just snow-covered areas)
    [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif', bands='all'))
    x[nosnow]=0 #set all 0m snow pixels to 0 density
    nans=np.where(x<0)
    x[nans]=0 #set all nans to 0
    bsl=np.where(SFAs[n]==1)
    x[bsl]=0 #set all pixels below snowline to 0
    Density.append(x)
del x,n,bsl

# Produce SWE maps (in mm) ---------------------------------------------------------------------------------------
SWE=[]
for n in range(len(phases)):
    x=((Depth[n]*Density[n]))*1000
    SWE.append(x)
del x,n

# Calculations ---------------------------------------------------------------------------------------------------------------------
min_elevs=[] # These are here just for reference: the min elevations of each subbasin
max_elevs=[] # These are here just for reference: the max elevations of each subbasin
for a in range(len(subbasin)):
    SWE_sub=[]
    Depth_sub=[]
    Density_sub=[]
    for n in range(len(phases)):
        # Clip SnowDepth, SnowDensity, SWE map by subbasin:
        if lakemodel == 'Y':
            [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution2)+'m/'+str(subbasin[a])+'_watershed_'+str(resolution2)+'m.tif'))
        else:
            [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution2)+'m/'+str(subbasin[a])+'_watershed_no_lakes_'+str(resolution2)+'m.tif'))
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
        x=np.nansum(SWE_sub[n]*int(resolution2)*int(resolution2))/1000
        total_SWV.append(x)
        y=np.nanmean(SWE_sub[n])
        mean_SWE_depth.append(y)
    del n,x,y

    # Calculate total snow volume, clip out areas below snowline
    total_SV=[]
    for n in range(len(phases)):
        x=Depth_sub[n]
        z=(np.nansum(Depth_sub[n])*int(resolution2)*int(resolution2))
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
            total_SWE=np.nansum(SWE_within_each_band[k]*int(resolution2)*int(resolution2))/1000
            total_SWV_banded.append(total_SWE) 
        banded_mean_SWE.append(mean_SWE_banded)
        banded_total_SWV.append(total_SWV_banded)

    # Set new elev bands for table
    min_elev2=min_elev+50
    max_elev2=max_elev-50
    elev_bands=np.arange(start=min_elev2, stop=max_elev2, step=100)

    # Output ------------------------------------------------------------------------------------------
    # Export SWE maps. Default is to only save a SWE map for the entire extent. Uncomment lines below if you want to save separate maps for subbasins.
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SWE/resolution_'+str(resolution2)+'m')
    for n in range(len(phases)):
        pyrsgis.raster.export(SWE[n], R, filename=str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SWE_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif')
    #if subbasin[a]!=extent:
    #    for n in range(len(phases)):
    #          pyrsgis.raster.export(SWE_sub[n], R, filename=str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_SWE_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif')

    # Export key numbers 
    path = str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Key_numbers/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)
    os.makedirs(path, exist_ok=True)
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Key_numbers/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'/')
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
    path = str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)
    os.makedirs(path, exist_ok=True)
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'/')
    for m in range(len(banded_total_SWV)):
        y=np.array(banded_total_SWV[m])
        y=y.reshape(len(y),)
        z=np.transpose(np.array([elev_bands,y]))
        zz=pd.DataFrame(z,columns=['elev_band','Total_SWV_m3'])
        zz.to_csv(str(subbasin[a])+'_'+str(year)+'_'+str(phases[m])+'_Elevation_banded_total_SWV.csv')
            
    # Save elevation-banded mean SWE
    for m in range(len(banded_mean_SWE)):
        y=np.array(banded_mean_SWE[m])
        y=y.reshape(len(y),)
        z=np.transpose(np.array([elev_bands,y]))
        zz=pd.DataFrame(z,columns=['elev_band','SWE_mean_mm'])
        zz.to_csv(str(subbasin[a])+'_'+str(year)+'_'+str(phases[m])+'_Elevation_banded_mean_SWE.csv')
    min_elevs.append(min)
    max_elevs.append(max)

# Save processing variables
var.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Metadata/'+str(extent)+'_'+str(year)+'_processing_variables_'+str(date)+'.csv', index = False)
