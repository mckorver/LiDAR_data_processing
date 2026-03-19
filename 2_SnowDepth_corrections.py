# This code:
# clips out unrealistic negative values (<-5m), and noise introduced by vegetation misclassification 
# caps values >-5 & <0 to 0m and unrealistic high values to 10m + mean snowdepth
# bias correction: add or subtract a uniform layer of snow
# clips out lakes and glaciers
# Outputs: 
# processed snowdepths (capped and clipped)
# processed snowdepths (capped, clipped, and vegetation corrected)

# Import package
import pyrsgis
import numpy as np
import pandas as pd
import os
import cv2

# Import input data --------------------------------------------------------------------------------
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
phases = []
bias_correction_snow = []
avalanche_threshold = []
upper_detection_threshold = []
lower_detection_threshold = []
kernel_size = []
expansion_distance = []
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if isinstance(y, float):
            a.append(int(y))
        else:
            a.append(y)
append_fun(phases,'phases')
append_fun(bias_correction_snow,'bias_correction_snow')
append_fun(avalanche_threshold,'avalanche_threshold')
append_fun(upper_detection_threshold,'upper_detection_threshold')
append_fun(lower_detection_threshold,'lower_detection_threshold')
append_fun(kernel_size,'kernel_size')
append_fun(expansion_distance,'expansion_distance')

# Import input data -----------------------------------------------------------
# Import provisional snow depths
SDs=[]
for n in range(len(phases)):
    [R,SD]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_'+str(resolution1)+'m.tif', bands='all'))
    SDs.append(SD)

# Add or subtract layer of snow to entire extent to account for bias
for n in range(len(phases)):
    x=SDs[n]
    y=np.where(x>0)
    x[y]=x[y] + bias_correction_snow[n]
    SDs[n]=x

# Import lakes (and glaciers) mask
if glaciers == 'Y':
    [R,lakes_glaciers]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_lakes_glaciers_'+str(resolution1)+'m.tif', bands='all'))
elif glaciers == 'N':
    [R,lakes_glaciers]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_lakes_'+str(resolution1)+'m.tif', bands='all'))

# Import bare earth and create avalanche mask
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))
avalanche_elevs=[]
for n in range(len(phases)):
    x=np.where(BE>avalanche_threshold[n])
    avalanche_elevs.append(x)
del BE

# Calculations -----------------------------------------------------------
# Set nans
for n in range(len(phases)):
    x=SDs[n]
    i=np.where(x<-100)
    x = x.astype(float)  # Convert integer array to float
    x[i] = np.nan
    SDs[n]=x
del x,n

# Remove lakes (and glaciers)
lakes_glaciers[lakes_glaciers<=0]= np.nan
i=np.where(lakes_glaciers==1)
for n in range(len(phases)):
    SD=SDs[n]
    SD[i]= np.nan
    SDs[n]=SD
del n,i,SD

# Cap snow depths greater than 10m above the mean or less than 0m and set values less than -5m and more than 30m to nan (assumed clouds)
for n in range(len(SDs)):
    x=SDs[n]
    mean=np.nanmean(x)
    sd=np.nanstd(x)
    i=np.where(x>=30)
    x[i]=np.nan
    j=np.where(x<-5)
    x[j]=np.nan
    k=np.where((x>(mean+10))&(x<30))
    x[k]=mean+10
    l=np.where((x<0)&(x>=-5))
    x[l]=0
    SDs[n]=x
del n,x,mean,sd,i,j

# Run vegetation correction
for n in range(len(phases)):
    SD=SDs[n]
    
    # Apply filter to produce smoothed rasters
    kernel=np.ones((kernel_size[n],kernel_size[n]),np.float32)/(kernel_size[n]*kernel_size[n])
    V=SD.copy()
    V[np.isnan(V)]=0
    V=cv2.filter2D(V,-1,kernel)
    W=0*SD.copy()+1
    W[np.isnan(W)]=0
    W=cv2.filter2D(W,-1,kernel)
    smooth=V/W
    smooth[np.isnan(SD)]=np.nan
    del V,W,kernel
    
    # Subtract smoothed DEM from original DEM
    diff=SD-smooth
    del smooth
    
    # Set and apply threshold for detection of bumps in subtracted rasters
    diff[diff<upper_detection_threshold[n]]=0
    diff[diff>upper_detection_threshold[n]]=1
    diff[diff<lower_detection_threshold[n]]=1
    veg_removal_mask=diff
    del diff
    
    # Expand anomylous areas to incorporate surrounding pixels
    kernel = np.ones((expansion_distance[n],expansion_distance[n]), np.uint8)
    expanded_mask=cv2.dilate(veg_removal_mask,kernel,iterations=1)
    del kernel,veg_removal_mask
    
    # Exclude areas where avalanches are likely to occur
    avalanche_corrected_mask=np.copy(expanded_mask)
    avalanche_corrected_mask[avalanche_elevs[n]]=0
    del expanded_mask
    
    # Remove misclassified areas from original snow depth DEMs
    SD_corrected=np.copy(SD)
    SD_corrected[avalanche_corrected_mask==1]=np.nan
    
    # Output -----------------------------------------------------------------------------------------------------------------------
    # Export capped and clipped snow depth maps
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m')
    pyrsgis.export(SDs[n],R,filename='Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_'+str(resolution1)+'m.tif')

    # Export capped, clipped, and vegcorrected snow depth maps
    pyrsgis.export(SD_corrected,R,filename='Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_vegcorrected_'+str(resolution1)+'m.tif')
    del SD_corrected

# Save processing variables
var.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Metadata/'+str(extent)+'_'+str(year)+'_processing_variables_'+str(date)+'.csv', index = False)
