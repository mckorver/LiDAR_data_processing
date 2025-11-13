# This script calculates the uncertainty in TWV
# This script outputs:
# a csv file with TWV errors by survey phase
# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

# 'CRU','Comox','Eric','Moat','Rees','Residual'
# 'MV','SEY','CAP','BurwellLake','LochLomond','PalisadeLake','UpperSeymour'
# 'ENG','Arrowsmith','Fishtail','Cokely'

# NOTE if running data from 2020-2024:
# Line 49: load 'SnowDepth_belowSL' SnowDepth maps, because in the final maps, data below the snowline was clipped out.
# Line 50-51: set the -1 data to 0

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin=['CRU','Comox','Eric','Moat','Rees','Residual'] #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2025' # Enter year of interest
phases=['P1','P2','P3'] # Enter survey phases ('P1','P2', etc.) Note run all surveys of a year simultaneously
BEversion = 2 # Enter Bare Earth version number
resolution = 2 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
lakemodel = 'Y' # Enter 'Y' or 'N' for including modelled SnowDepth on lakes
glaciermodel = 'Y' # Enter 'Y' or 'N' for including a SWE model for glaciers
rand_model_error=72 # ENTER RANDOM DENSITY MODEL ERROR (kg/m3) Cruikshank = 72, Englishman = 56, Metro Vancouver = 37, Tsitika = 51

# Import packages
import numpy as np
import pyrsgis
import os
import pandas as pd
from pathlib import Path

# Import input data -------------------------------------------------------------------------
# Import watershed mask (without lakes or glaciers)
if lakemodel == 'Y' and glaciermodel == 'Y':
    [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_'+str(resolution)+'m.tif'))
elif lakemodel == 'N' and glaciermodel == 'Y':
    [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_no_lakes_'+str(resolution)+'m.tif'))
elif lakemodel == 'N' and glaciermodel == 'N':
    [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(watershed)+'_watershed_no_lakes_no_glaciers_'+str(resolution)+'m.tif'))
nans=np.where(WS<1)
WS[nans]=np.nan

# Import snow density data (g/cm3 --> converted to kg/m3) and clip to WS
D=[]
for n in range(len(phases)):
    if lakemodel == 'Y':
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodelY.tif', bands='all'))
    else:
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodelN.tif', bands='all'))
    x[x<0]=np.nan
    x=x*1000*WS
    D.append(x)
del n,x

# Import snow depth data (m) and clip to WS
SD=[]
for n in range(len(phases)):
    if lakemodel == 'Y':
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodelY.tif', bands='all'))
    else:
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodelN.tif', bands='all'))
    x = x.astype(float) # Convert to float to allow NaNs
    x[x < 0] = np.nan
    x=x*WS
    SD.append(x)
del x,n

# Import bare earth and clip to WS
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
BE[BE<0]=np.nan
BE=BE*WS

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

# Calculations -----------------------------------------------------------------------------        
# Extract snow free areas from snow depth maps
SF_SD=[]
for n in range(len(SD)):
    i=np.where((SFAs[n]==1)&(SD[n]/SD[n]==1))
    x=SD[n]
    y=x[i]
    SF_SD.append(y)
del x,n,y
    
# Find mean and standard deviation in snow free elevation differences (systematic and random errors in snow depth)
syst_depth_error=[]
rand_depth_error=[]
for n in range(len(SF_SD)):
    x=np.nanmedian(SF_SD[n])
    syst_depth_error.append(x)
    y=np.nanstd(SF_SD[n])
    rand_depth_error.append(y)
del n,x,y

# Find covariance between density and depth
covariances=[]
for n in range(len(SD)):
    mean_SD=np.nanmean(SD[n])
    mean_D=np.nanmean(D[n])
    covariance=(np.nansum((SD[n]-mean_SD)*(D[n]-mean_D)))/len(SD[n])
    covariances.append(covariance)
del mean_SD,mean_D,covariance,n

# Create distributed error maps for error propagation purposes
syst_depth_error_maps=[]
rand_depth_error_maps=[]
for n in range(len(SD)):
    x=(WS/WS)*syst_depth_error[n]
    syst_depth_error_maps.append(x)
    y=(WS/WS)*rand_depth_error[n]
    rand_depth_error_maps.append(y)   
del n,x,y

# Find per pixel snow volume errors (m3)
syst_volume_error=[]
rand_volume_error=[]
for n in range(len(syst_depth_error_maps)):
    x=syst_depth_error_maps[n]*resolution*resolution
    syst_volume_error.append(x)
    y=rand_depth_error_maps[n]*resolution*resolution
    rand_volume_error.append(y)
del n,x,y

# Calculate random snow density error (kg/m3)
rand_density_error=[]
for n in range(len(D)):
    x=rand_model_error
    rand_density_error.append(x)
del n,x

# Distribute random snow density error
rand_density_error_maps=[]
for n in range(len(rand_density_error)):
    x=(SD[n]/SD[n])*rand_density_error[n]
    rand_density_error_maps.append(x)
del n,x

# Calculate snow volume within each pixel (m3)
snow_volume=[]
for n in range(len(SD)):
    x=SD[n]*resolution*resolution
    snow_volume.append(x)
del n,x

# Calculate systematic snow mass uncertainty (m3*kg/m3 = kg)
syst_mass_error=[]
for n in range(len(syst_depth_error_maps)):
    x=syst_volume_error[n]*D[n]
    syst_mass_error.append(x)
del n,x

# Calculate random snow mass uncertainty (kg)
rand_mass_error=[]
for n in range(len(rand_depth_error_maps)):
    x=np.sqrt(((rand_density_error_maps[n]*snow_volume[n])**2)+((rand_volume_error[n]*D[n])**2)+(2*snow_volume[n]*D[n]*covariances[n]))
    rand_mass_error.append(x)
del n,x

# Calculate overall per pixel mass uncertainty (kg)
mass_error=[]
for n in range(len(rand_mass_error)):
    x=np.sqrt((rand_mass_error[n]**2)+(syst_mass_error[n]**2))
    mass_error.append(x)
del n,x

# Find snow mass distribution and total snow mass in basin (kg)
snow_mass=[]
total_snow_mass=[]
for n in range(len(SD)):
    x=SD[n]*D[n]*resolution*resolution
    y=np.nansum(x)
    snow_mass.append(x)
    total_snow_mass.append(y)
del n,x

# Calculate basin-wide systematic snow mass uncertainty (kg)
basin_syst_mass_error=[]
for n in range(len(syst_mass_error)):
    x=np.nansum(syst_mass_error[n])
    basin_syst_mass_error.append(x)
del n,x

# Calculate basin-wide random snow mass uncertainty (kg)
basin_rand_mass_error=[]
for n in range(len(rand_mass_error)):
    x=np.sqrt(np.nansum(rand_mass_error[n]**2))
    basin_rand_mass_error.append(x)
del n,x

# Calculate overall snow mass uncertainty 
basin_mass_error=[] # kg = l = 0.001 m3
basin_water_volume_error=[] # m3
for n in range(len(rand_mass_error)):
    x=np.sqrt((basin_rand_mass_error[n]**2)+(basin_syst_mass_error[n]**2))
    y=x/1000
    basin_mass_error.append(x)
    basin_water_volume_error.append(y)
del n,x

# Calculate percentage basin mass error
percentage_basin_mass_error=[]
for n in range(len(basin_mass_error)):
    x=(basin_mass_error[n]/total_snow_mass[n])*100
    percentage_basin_mass_error.append(x)
del n,x

# Output -------------------------------------------------------------------------------
# Export results
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Key_numbers/'+str(year)+'/resolution_'+str(resolution)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/')
percentage_basin_mass_error1=pd.DataFrame(list(zip(phases,percentage_basin_mass_error)),columns=['Survey','Percentage_total_SWV_errors'])
for a in range(len(subbasin)):
    percentage_basin_mass_error1.to_csv(str(subbasin[a])+'_'+str(year)+'_Percentage_total_SWV_errors.csv')
basin_water_volume_error1=pd.DataFrame(list(zip(phases,basin_water_volume_error)),columns=['Survey','Absolute_total_SWV_errors_m3'])
basin_water_volume_error1.to_csv(str(watershed)+'_'+str(year)+'_Absolute_total_SWV_errors.csv')
del D,i,mass_error,rand_density_error_maps,rand_depth_error_maps,rand_mass_error,rand_volume_error,SD,SF_SD,snow_mass,snow_volume,syst_depth_error_maps,syst_mass_error,syst_volume_error,total_snow_mass