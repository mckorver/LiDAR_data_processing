# This script calculates snowdensity with a machine learning random forest model based on
# in-situ snowdensity & snowdepth, elevation (snow-free bare earth), slope, northness, eastness, curvature, and Xt.
# NOTE that this script uses a model that does not take canopy cover and canopy height as input parameters 
# This script outputs:
# A snowdensity raster map

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='MV' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='UpperSeymour' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2025' # Enter year of interest
phase='P3' # Enter survey phase. NOTE only one phase can be run at the time in this script
BEversion = 6 # Enter Bare Earth version number
resolution = 1 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone

import numpy as np
import pyrsgis
import os
import joblib
import pandas as pd
import gc
from pathlib import Path

# Import input data -----------------------------------------------------------------------------------------------------
# Import snow density model
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Overall_density_model/No_Canopy')
rf = joblib.load('RF_density_model_'+str(watershed)+'_NoCan_2021_2024.joblib')
all_scalers=[]
for n in range(1,8):
    scaler = joblib.load('scaler'+str(n)+'.pkl')
    all_scalers.append(scaler)
del scaler

# Import model input data
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/v'+str(BEversion)+'/resolution_'+str(resolution)+'m')
SurfaceCharFiles=[str(subbasin)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif',str(subbasin)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif']
input=[]
for n in range(len(SurfaceCharFiles)):
    [R,x]=np.array(pyrsgis.raster.read(SurfaceCharFiles[n], bands='all'))
    input.append(x)
rows=x.shape[0]
cols=x.shape[1]
# Set nan values
for n in range(len(input)):
    x=input[n]
    x[x<-1]='nan'
    input[n]=x
del SurfaceCharFiles,x,n

# Import Xt data
[R,Xt]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution)+'m/Distributed_Xt_'+str(subbasin)+'_'+str(year)+'_'+str(phase)+'.tif', bands='all'))
x=Xt
x[x<0]='nan'
Xt=x
Xt=np.ndarray.flatten(Xt)
del x

# Import snow depth data (convert to cm)
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDepth/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/')
[R,SD]=np.array(pyrsgis.raster.read(str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SnowDepth.tif', bands='all'))
nans=np.where(SD<-100)
SD[nans]='nan'
SD=np.ndarray.flatten(SD)*100 # convert to cm

# Import bare earth
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]='nan'
BE=BE.astype('float64')
del nans

# Import snow free areas. A raster mask is used if available, otherwise an elevation threshold.
file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SFA.tif')
if file.is_file():
    [R,SFA]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SFA.tif', bands='all'))
else:
    x=pd.read_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Snow_free_elevation_masks/SFETs_'+str(subbasin)+'_'+str(year)+'.csv')
    SFET=x[x['Survey']==phase]
    SFET=SFET['SFETs'].iloc[0]
    SFA=BE/BE
    i=np.where(BE<SFET)
    j=np.where(BE>=SFET)
    SFA[i]=1
    SFA[j]=0

# Set NaNs at SFA for all input data
x=SD
y=np.ndarray.flatten(SFA)
nans=np.where(y==1)
x[nans]='nan'
SD=x

x=Xt
y=np.ndarray.flatten(SFA)
nans=np.where(y==1)
x[nans]='nan'
x=x.reshape(len(x),1)
Xt=x

Surface_Characteristics=[]
for n in range(len(input)):
    x=input[n]
    x=np.ndarray.flatten(x)
    y=np.ndarray.flatten(SFA)
    nans=np.where(y==1)
    x[nans]='nan'
    x=x.reshape(len(x),1)
    Surface_Characteristics.append(x)
del n,x,y

# Organise input data for each survey
input_parameters=[SD.astype('float64'),Surface_Characteristics[0].astype('float64'),Surface_Characteristics[1].astype('float64'),Surface_Characteristics[2].astype('float64'),Surface_Characteristics[3].astype('float64'),Xt.astype('float64'),Surface_Characteristics[4].astype('float64')]
del Surface_Characteristics,Xt

# Normalise input variables
for n in range(len(input_parameters)):   
    x=input_parameters[n]
    x=x.reshape(len(x),1)
    scaler = all_scalers[n]
    X_normalized = scaler.transform(x)
    X_normalized=X_normalized.reshape(len(X_normalized),)
    x=X_normalized
    input_parameters[n]=x  
del x,scaler,all_scalers,X_normalized

# Set nans to 0 in input parameters, and get indices of nans
all_nans_all_surveys=[]
for n in range(len(input_parameters)):
    ip=input_parameters[n]
    all_nans = np.nonzero(np.isnan(ip))
    ip[all_nans]=0
    all_nans_all_surveys.append(all_nans)
    input_parameters[n]=ip
del n,ip,nans,all_nans   

# Reformat model input data
for n in range(len(input_parameters)):
    x=input_parameters[n]
    y=x.reshape(len(x),1)
    input_parameters[n]=y
input_parameters=np.concatenate(input_parameters,axis=1)
del n,x,y
gc.collect()

# Run snow density model
Simulated_density=rf.predict(input_parameters)
del input_parameters

# Create mask for extent of snowdepth data
study_area=SD
i=np.where(SD<0)
j=np.where(SD>=0)
study_area[i]='nan'
study_area[j]=1

# Set the pixels below snow cover and outside the data outline to nans and reshape into 2-dimensional arrays
Simulated_density=Simulated_density*study_area
Simulated_density=np.reshape(Simulated_density,(rows,cols))
del all_nans_all_surveys,rows,cols

# Output ------------------------------------------------------------------------------------------
# Export simulated snow density map
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m')
pyrsgis.raster.export(Simulated_density, R, filename=str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SnowDensity.tif')

# OPTIONAL adjust density values
#x=Simulated_density
#y=np.where((x>0)&(x<0.45))
#x[y]=x[y] + 0.02
#Simulated_density=x

#os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SnowDensity/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m')
#pyrsgis.raster.export(Simulated_density, R, filename=str(subbasin)+'_'+str(year)+'_'+str(phase)+'_SnowDensity.tif')
