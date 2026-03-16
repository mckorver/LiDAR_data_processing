# This code loads LiDAR data, calculates snow depths, and performs an initial validation on snow-free roads
# Outputs: 
# unprocessed snowdepths
# snow-free road bias analysis (histogram, mean, median)

# Import package
import pyrsgis
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import input data --------------------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/input_data/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
BEversion = var['BEversion'][0]
phases = []
x = var['phases'][var['phases'].notna()]
for n in range(len(x)):
    a = x[n]
    phases.append(a)

# Import snow surface elevations
SSEs=[]
for n in range(len(phases)):
    [R,x]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/input_data/LiDAR_data/'+str(watershed)+'/'+str(year)+'/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SS.tif', bands='all'))
    SSEs.append(x)
del x,n

# Import bare earth
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))

# Import roads
[R,ROADS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Roads_mask/resolution_'+str(resolution1)+'m/'+str(extent)+'_ROADS.tif', bands='all'))

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

# Calculations ---------------------------------------------------------------------------------------------------------------
# Set nans
BE[BE<0]= np.nan
ROADS[ROADS<=0] = np.nan
for n in range(len(phases)):
    x=SSEs[n]
    i=np.where(x<-100)
    x[i]=np.nan
    SSEs[n]=x
    x=SFAs[n]
    i=np.where(x<0)
    x[i]=np.nan
    SFAs[n]=x
del x,n

# Calculate snow depths
SDs=[]
for n in range(len(phases)):
    x=SSEs[n]-BE
    SDs.append(x)
del x,n

# Extract snow free roads from snow depth maps
SDs_SFRs=[]
for n in range(len(phases)):
    x=SDs[n]*SFAs[n]*ROADS
    nans=np.where(x==0)
    x[nans]=np.nan
    std3=np.nanstd(x)*3
    x = x[x>-std3]
    x = x[x<std3]
    SDs_SFRs.append(x)
del x,n

# Calculate biases on snow free roads
SFR_biases_mean=[]
SFR_biases_median=[]
SFR_biases_std=[]
for n in range(len(SDs_SFRs)):
    x=np.nanmean(SDs_SFRs[n])
    y=np.nanmedian(SDs_SFRs[n])
    z=np.nanstd(SDs_SFRs[n])
    SFR_biases_mean.append(x)
    SFR_biases_median.append(y)
    SFR_biases_std.append(z)
del x,y,z,n

# Extract snow free roads from BE maps
BE_road=[]
SSE_road=[]
for n in range(len(phases)):
    x=BE*SFAs[n]*ROADS
    a=SSEs[n]*SFAs[n]*ROADS
    nansx=np.where(x==0)
    x[nansx]=np.nan
    std3=np.nanstd(x)*3
    x = x[x>-std3]
    x = x[x<std3]
    nansa=np.where(a==0)
    a[nansa]=np.nan
    std3=np.nanstd(a)*3
    a = a[a>-std3]
    a = a[a<std3]
    BE_road.append(x)
    SSE_road.append(a)
del x,a,nansx,nansa,n

# Output ---------------------------------------------------------------------------------------------------------------------------
# Plot histogram of snow depth values along snow free roads
# Mean and std snow depth is shown as red vertical lines, to compare to black zero line
for n in range(len(phases)):
    x = SDs_SFRs[n]
    x = x[~np.isnan(x)]          # remove NaNs
    counts = x
    # bins of width 0.05 from -0.5 to 0.5
    bins = np.arange(-0.5, 0.5 + 0.05, 0.01)
    plt.hist(counts, bins, label=phases[n], edgecolor='black', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='-', linewidth=1.5)
    plt.axvline(SFR_biases_mean[n], color='r', linestyle='-', linewidth=1.5)
    plt.axvline(SFR_biases_mean[n]-SFR_biases_std[n], color='r', linestyle='--', linewidth=1)
    plt.axvline(SFR_biases_mean[n]+SFR_biases_std[n], color='r', linestyle='--', linewidth=1)
    plt.legend(loc='upper right')
    plt.ylabel('Count')
    plt.xlabel('Snow depths (m) along snow free roads')
    plt.savefig(
        f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
        f"Snow_free_road_histogram_{extent}_{year}_{phases[n]}.png")
    plt.close()
    del x, counts, bins

# Export biases into csv file
for n in range(len(phases)):
    SFR_biases=np.array([phases[n],SFR_biases_mean[n],SFR_biases_median[n],SFR_biases_std[n]]).reshape(1,4)
    biases_reformatted=pd.DataFrame(data=SFR_biases,columns=['Survey','Mean_bias','Median_bias','Std_bias'])
    biases_reformatted.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/Snow_free_road_biases_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'.csv', index=False)

# Save snow depth maps
for n in range(len(phases)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m')
    pyrsgis.export(SDs[n],R,filename='Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_'+str(resolution1)+'m.tif')

# Save processing variables
var.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Metadata/'+str(extent)+'_'+str(year)+'_processing_variables.csv')
