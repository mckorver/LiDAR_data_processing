# This code:
# clips out noisy areas (visually identified), lakes, unrealistic negative values (<-5m), and noise introduced by vegetation misclassification 
# caps values >-5 & <0 to 0m and unrealistic high values to 10m + mean snowdepth
# bias correction: add or subtract a uniform layer of snow
# clips out lakes and glaciers
# Outputs: 
# processed snowdepths (capped and clipped)
# processed snowdepths (capped, clipped, and vegetation corrected)

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='CRU' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2025' # Enter year of interest
phases=['P1','P2','P3'] # Enter survey phases ('P1','P2', etc.)
BEversion = 2 # Enter Bare Earth version number.
resolution = 1 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
glaciers = 'Y' # Enter 'Y' if the watershed has glaciers, 'N' if not

bias_correction=[0.06,0.12,0.07] # For every phase: set bias correction (in metres)
avalanche_threshold=[750,750,750] # For every phase: set highest elevation below which no avalanches are visible (in metres)
upper_detection_threshold=[4,4,4] # For every phase: set upper threshold of snow depth anomalies to be removed (in metres)
lower_detection_threshold=[-2,-2,-2] # For every phase: set lower threshold of snow depth anomalies to be removed (in metres)
kernel_size=[200,200,200] # For every phase: set kernel size for filter used to detect anomalies (in pixels)
expansion_distance=[10,10,10] # For every phase: set distance used to expand anomylous areas to be removed (in pixels)

# Import package
import pyrsgis
import numpy as np
import pandas as pd
import os
import cv2
from pathlib import Path

# Import input data -----------------------------------------------------------
# Import provisional snow depths
SDs=[]
for n in range(len(phases)):
    [R,SD]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m/Provisional_SD_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_'+str(resolution)+'m.tif', bands='all'))
    SDs.append(SD)

# Add or subtract layer of snow to entire extent to account for bias
for n in range(len(phases)):
    x=SDs[n]
    y=np.where(x>0)
    x[y]=x[y] + bias_correction[n]
    SDs[n]=x

# Optional: Import peripheral masks for areas around the periphery of AOI where data is poor quality
masks=[]
for n in range(len(phases)):
    file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Peripheral_masks/'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_PeriMask.tif')
    if file.is_file():
        [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Peripheral_masks/'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_PeriMask.tif', bands='all'))
    else:
        x=[]
    masks.append(x)
del x,n

# Import lakes (and glaciers) mask
if glaciers == 'Y':
    [R,lakes_glaciers]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/'+str(subbasin)+'_lakes_glaciers_'+str(resolution)+'m.tif', bands='all'))
elif glaciers == 'N':
    [R,lakes_glaciers]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Lakes_and_glaciers_mask/resolution_'+str(resolution)+'m/'+str(subbasin)+'_lakes_'+str(resolution)+'m.tif', bands='all'))

# Import bare earth and create avalanche mask
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
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
lakes_glaciers[lakes_glaciers==0]= np.nan
i=np.where(lakes_glaciers==1)
for n in range(len(phases)):
    SD=SDs[n]
    SD[i]= np.nan
    SDs[n]=SD
del n,i,SD

# Remove peripheral areas
for n in range(len(phases)):
    mask=masks[n]
    i=np.where(mask==1)
    SD=SDs[n]
    SD[i]=np.nan
    SDs[n]=SD
del n,i,mask,SD

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
    #plt.figure(dpi=1000)
    #plt.imshow(smooth,vmin=0,vmax=5)
    #plt.colorbar()
    del V,W,kernel
    
    # Subtract smoothed DEM from original DEM
    diff=SD-smooth
    #plt.figure(dpi=1000)
    #plt.imshow(diff,vmin=0,vmax=5)
    #plt.colorbar()
    del smooth
    
    # Set and apply threshold for detection of bumps in subtracted rasters
    diff[diff<upper_detection_threshold[n]]=0
    diff[diff>upper_detection_threshold[n]]=1
    diff[diff<lower_detection_threshold[n]]=1
    veg_removal_mask=diff
    #plt.figure(dpi=1000)
    #plt.imshow(veg_removal_mask,vmin=0,vmax=1)
    #plt.colorbar()
    del diff
    
    # Expand anomylous areas to incorporate surrounding pixels
    kernel = np.ones((expansion_distance[n],expansion_distance[n]), np.uint8)
    expanded_mask=cv2.dilate(veg_removal_mask,kernel,iterations=1)
    #plt.figure(dpi=1000)
    #plt.imshow(expanded_mask,vmin=0,vmax=1)
    del kernel,veg_removal_mask
    
    # Exclude areas where avalanches are likely to occur
    avalanche_corrected_mask=np.copy(expanded_mask)
    avalanche_corrected_mask[avalanche_elevs[n]]=0
    #plt.figure(dpi=1000)
    #plt.imshow(avalanche_corrected_mask,vmin=0,vmax=1)
    del expanded_mask
    
    # Remove misclassified areas from original snow depth DEMs
    SD_corrected=np.copy(SD)
    SD_corrected[avalanche_corrected_mask==1]=np.nan
    
    # Output -----------------------------------------------------------------------------------------------------------------------
    # Export capped and clipped snow depth maps
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m')
    pyrsgis.export(SDs[n],R,filename='Provisional_SD_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_'+str(resolution)+'m.tif')

    # Export capped, clipped, and vegcorrected snow depth maps
    pyrsgis.export(SD_corrected,R,filename='Provisional_SD_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_vegcorrected_'+str(resolution)+'m.tif')
    del SD_corrected

# Export vegetation correction variables used
for n in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Processing_variables/'+str(year))
    data=np.array([phases[n],bias_correction[n],avalanche_threshold[n],upper_detection_threshold[n],lower_detection_threshold[n],kernel_size[n],expansion_distance[n]]).reshape(1,7)
    df = pd.DataFrame(data=data, columns=['Survey','bias_correction','avalanche_threshold','upper_detection_threshold','lower_detection_threshold','kernel_size','expansion_distance'])
    df.to_csv(str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_vegcorrection_variables.csv',index=False)