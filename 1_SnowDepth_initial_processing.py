# This code loads LiDAR data, calculates snow depths, and checks the bias on snow-free roads
# Outputs: 
# unprocessed snowdepths
# bias analysis (histogram, mean, median)
#%%
# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='CRU' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2025' # Enter year of interest
phases=['P1','P2','P3'] # Enter survey phases ('P1','P2', etc.)
BEversion = 2 # Enter Bare Earth version number.
resolution = 1 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone

# Import package
import pyrsgis
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import input data ---------------------------------------------------
# Import snow surface elevations
SSEs=[]
for n in range(len(phases)):
    [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/input_data/LiDAR_data/'+str(watershed)+'/'+str(year)+'/'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_SS.tif', bands='all'))
    SSEs.append(x)
del x,n

# Import bare earth
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))

# Import roads
[R,ROADS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Roads_mask/'+str(subbasin)+'_ROADS.tif', bands='all'))

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

# Set nans
BE[BE < 0]= np.nan
ROADS = ROADS.astype(float)  # Convert from int to float
ROADS[ROADS < 0] = np.nan    # Set negative values to NaN
for n in range(len(phases)):
    x=SSEs[n]
    i=np.where(x<-100)
    x[i]='nan'
    SSEs[n]=x
    x=SFAs[n]
    i=np.where(x<=0)
    x[i]='nan'
    SFAs[n]=x
del x,n

# Calculate snow depths
SDs=[]
for n in range(len(phases)):
    x=SSEs[n]-BE
    SDs.append(x)
del x,n,SSEs

# Extract snow free roads from snow surface elevation maps
SDs_SFRs=[]
for n in range(len(phases)):
    x=SDs[n]*SFAs[n]*ROADS
    nans=np.where(x==0)
    x[nans]='nan'
    SDs_SFRs.append(x)
del x,ROADS,SFAs,n

# Calculate biases on snow free roads
SFR_biases_mean=[]
SFR_biases_median=[]
for n in range(len(SDs_SFRs)):
    x=np.nanmean(SDs_SFRs[n])
    y=np.nanmedian(SDs_SFRs[n])
    SFR_biases_mean.append(x)
    SFR_biases_median.append(y)
del x,y,n   

# Output -----------------------------------------------------------------
# Plot histogram of snow depth values along snow free roads
#counts=[]
for n in range(len(phases)):
    x = SDs_SFRs[n]
    x = x[~np.isnan(x)]          # remove NaNs
    counts = x
    # bins of width 0.05 from -0.5 to 0.5
    bins = np.arange(-0.5, 0.5 + 0.05, 0.05)
    plt.hist(counts, bins, label=phases[n], edgecolor='black', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.legend(loc='upper right')
    plt.ylabel('Count')
    plt.xlabel('Snow depths (m) along snow free roads')
    plt.savefig(
        f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
        f"Snow_free_road_histogram_{subbasin}_{year}_{phases[n]}.png")
    plt.close()
    del x, counts, bins
# Export biases into csv file
for n in range(len(phases)):
    SFR_biases=np.array([phases[n],SFR_biases_mean[n],SFR_biases_median[n]]).reshape(1,3)
    biases_reformatted=pd.DataFrame(data=SFR_biases,columns=['Survey','Mean_bias','Median_bias'])
    biases_reformatted.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/Snow_free_road_biases_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.csv', index=False)

# Save snow depth maps
for n in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution)+'m')
    pyrsgis.export(SDs[n],R,filename='Provisional_SD_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_'+str(resolution)+'m.tif')
    print(n+1)
