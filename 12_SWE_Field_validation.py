# This script validates the snow depth, snow density, and SWE LiDAR derived data to field data
# This script outputs:
# csv files of plot averaged statistics (mean, sd) and difference (LiDAR - field) statistics
# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

#This script performs validation checks 

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='MV' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin = 'MV'
year='2025' # Enter year of interest
phases=['P1','P2','P3'] # Enter survey phases ('P1','P2', etc.) NOTE run all surveys of a year simultaneously
resolution = 2 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
lakemodel = 'Y' # Enter 'Y' or 'N' for including modelled SnowDepth on lakes
glaciermodel = 'N' # Enter 'Y' or 'N' for including a SWE model for glaciers

import rasterio
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Import input data ------------------------------------------------------------------
# Read field data collection coordinates
os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/'+str(year))
#filenames=sorted(glob.glob('**.csv'))
field_data=[]
for n in range(len(phases)):
    x=pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[n]+'.csv'))
    #x = x[x['sample_rating']>data_quality]
    field_data.append(x)
del n,x

SWE_ids=[]
SWE_eastings=[]
SWE_northings=[]
FieldSWEs=[]
for n in range(len(field_data)):
    x = field_data[n]
    y = x[x['swe_final'].notnull()]
    plot_id=np.array(y['plot_id'])
    plot_id=plot_id.reshape(len(plot_id),)
    easting=np.array(y['easting_m']).astype('float64')
    easting=easting.reshape(len(easting),)
    northing=np.array(y['northing_m']).astype('float64')
    northing=northing.reshape(len(northing),)
    SWE=np.array(y['swe_final']).astype('float64')
    SWE=SWE.reshape(len(SWE),)
    SWE_ids.append(plot_id)
    SWE_eastings.append(easting)
    SWE_northings.append(northing)
    FieldSWEs.append(SWE)
del x,y,plot_id,easting,northing,SWE
                      
# Read gridded input datasets
LidarSWEs=[] #in mm
for n in range(len(phases)):
    if lakemodel == 'Y' and glaciermodel == 'Y':
        x=rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SWE/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_SWE_lakemodelY_glaciermodelY.tif')    
    elif lakemodel == 'N' and glaciermodel == 'Y':
        x=rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SWE/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_SWE_lakemodelN_glaciermodelY.tif')    
    elif lakemodel == 'N' and glaciermodel == 'N':
        x=rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Maps/SWE/'+str(watershed)+'/'+str(year)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'_SWE_lakemodelN_glaciermodelN.tif')
    LidarSWEs.append(x)
del n,x

# Sample gridded datasets at field data collection points
SWE_coord_list=[]
for m in range(len(phases)):
    coord_list=[]
    easting=SWE_eastings[m]
    northing=SWE_northings[m]
    for n in range(len(easting)):
        cl=(easting[n],northing[n])
        coord_list.append(cl)
    SWE_coord_list.append(coord_list)
LidarSWEs_sampled=[]
for n in range(len(phases)):
    x=[x for x in LidarSWEs[n].sample(SWE_coord_list[n])]
    x=np.array(x)
    LidarSWEs_sampled.append(x)
del n,m,SWE_coord_list,SWE_eastings,easting,northing,SWE_northings,cl,coord_list,x

##### Plot-averaged comparisons ---------------------------------------------------------------
FieldSWE_plot_mean=[]
FieldSWE_plot_sd=[]
FieldSWE_plot_names=[]
for n in range(len(phases)):
    fd=FieldSWEs[n]*10 #convert to mm
    fd=fd.reshape(len(fd),1)
    plot_id=SWE_ids[n]
    pp=[]
    for x in range(len(plot_id)):
        p=str(plot_id[x])
        pp.append(p)
    output_list = []
    for word in pp:
        if word not in output_list:
            output_list.append(word)
    b=[]
    a=[]
    pp=np.array(pp)
    for y in range(len(output_list)):
        i=np.where(pp==str(output_list[y]))
        ppp=fd[i]
        ppp = np.where(ppp == -9999, np.nan, ppp)  # convert -9999 to np.nan
        mean=np.nanmean(ppp)
        sd=np.nanstd(ppp)
        b.append(mean)
        a.append(sd)
    FieldSWE_plot_mean.append(b)
    FieldSWE_plot_sd.append(a)
    FieldSWE_plot_names.append(output_list)

LidarSWE_plot_mean=[]
LidarSWE_plot_sd=[]
LidarSWE_plot_names=[]
for n in range(len(phases)):
    fd=LidarSWEs_sampled[n]
    fd=fd.reshape(len(fd),1)
    plot_id=SWE_ids[n]
    pp=[]
    for x in range(len(plot_id)):
        p=str(plot_id[x])
        pp.append(p)
    output_list = []
    for word in pp:
        if word not in output_list:
            output_list.append(word)
    b=[]
    a=[]
    pp=np.array(pp)
    for y in range(len(output_list)):
        i=np.where(pp==str(output_list[y]))
        ppp=fd[i]
        ppp = np.where(ppp == -9999, np.nan, ppp)  # convert -9999 to np.nan
        mean=np.nanmean(ppp)
        sd=np.nanstd(ppp)
        b.append(mean)
        a.append(sd)
    LidarSWE_plot_mean.append(b)
    LidarSWE_plot_sd.append(a)
    LidarSWE_plot_names.append(output_list)

#### Difference (LiDAR - Field) statistics----------------------------------------------------------------------------------
# Calculate field-LiDAR differences
SWE_diffs=[]
SWE_meandiff=[]
SWE_sddiff=[]
SWE_rmsediff=[]
for n in range(len(LidarSWEs_sampled)):
    x=LidarSWEs_sampled[n].reshape(len(LidarSWEs_sampled[n],))
    y=(x-(FieldSWEs[n]*10)) #Convert fieldSWE in cm to mm
    z=np.nanmean(y)
    zz=np.nanstd(y)
    mse=(np.nansum(y**2))/len(y)
    rmse=math.sqrt(mse)
    SWE_diffs.append(y)
    SWE_meandiff.append(z)
    SWE_sddiff.append(zz)
    SWE_rmsediff.append(rmse)
del n,x,y,z,zz

# Calculate mean differences for each individual plot
SWE_plot_meandiffs=[]
SWE_plot_sddiffs=[]
SWE_plot_names=[]
for n in range(len(phases)):
    SWEdiff=SWE_diffs[n]
    SWEdiff=SWEdiff.reshape(len(SWEdiff),1)
    plot_id=SWE_ids[n]
    pp=[]
    for x in range(len(plot_id)):
        p=str(plot_id[x])
        pp.append(p)
    output_list = []
    for word in pp:
        if word not in output_list:
            output_list.append(word)
    SWE_plot_meandiff=[]
    SWE_plot_sddiff=[]
    pp=np.array(pp)
    for y in range(len(output_list)):
        i=np.where(pp==str(output_list[y]))
        ppp=SWEdiff[i]
        mean=np.nanmean(ppp)
        sd=np.nanstd(ppp)
        SWE_plot_meandiff.append(mean)
        SWE_plot_sddiff.append(sd)
    SWE_plot_meandiff=np.array(SWE_plot_meandiff)
    SWE_plot_meandiff=SWE_plot_meandiff.reshape(len(SWE_plot_meandiff),1)
    SWE_plot_sddiff=np.array(SWE_plot_sddiff)
    SWE_plot_sddiff=SWE_plot_sddiff.reshape(len(SWE_plot_sddiff),1)
    SWE_plot_meandiffs.append(SWE_plot_meandiff)
    SWE_plot_sddiffs.append(SWE_plot_sddiff)
    SWE_plot_names.append(output_list)

# Output ------------------------------------------------------------------------------------------------------------------
# Export difference statistics
# For the entire watershed
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/Difference_statistics/')
SWE_field=pd.DataFrame(list(zip(phases,SWE_meandiff,SWE_sddiff,SWE_rmsediff)),columns=['survey','SWE_mean_diff_m','SWE_sd_diff_m','SWE_rmse_diff_m'])
SWE_field.to_csv('Field_differences_SWE.csv', index=False)

# By plot
y = []
for n in range(len(phases)):
    surveys=np.repeat(phases,len(SWE_plot_names[n]))
    x=pd.DataFrame(list(zip(surveys,SWE_plot_names[n],SWE_plot_meandiffs[n].flatten(),SWE_plot_sddiffs[n].flatten())),columns=['survey','Plot_id','SWE_mean_diff_m','SWE_sd_diff_m'])
    y.append(x)
SWE_plot=pd.concat(y)
SWE_plot.to_csv('Plot_differences_SWE.csv', index=False)

# Export plot averaged statistics
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/Plot_averaged/')
y = []
for n in range(len(phases)):
    surveys=np.repeat(phases[n],len(FieldSWE_plot_names[n]))
    x=pd.DataFrame(list(zip(surveys,FieldSWE_plot_names[n],FieldSWE_plot_mean[n],FieldSWE_plot_sd[n],LidarSWE_plot_mean[n],LidarSWE_plot_sd[n])),columns=['survey','Plot_id','Field_SWE_mean','Field_SWE_sd','Lidar_SWE_mean','Lidar_SWE_sd'])
    y.append(x)
FieldSWE = pd.concat(y)
FieldSWE.to_csv('Plot_comparisons_SWE.csv', index=False)

g = sns.FacetGrid(FieldSWE, col='survey',hue='Plot_id')
def plot(x, y, xerr, yerr, **kwargs):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt = 'o', **kwargs)
    plt.grid(True)
g = g.map(plot, 'Field_SWE_mean', 'Lidar_SWE_mean', 'Field_SWE_sd', 'Lidar_SWE_sd')
def plot_one_to_one(x, y, **kwargs):
    ax = plt.gca()
    min_val = 0
    max_val = 1500
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--',**kwargs)
g = g.map(plot_one_to_one, 'Field_SWE_mean', 'Lidar_SWE_mean')
g.set_xlabels("Mean SWE (Field) [mm]")
g.set_ylabels("Mean SWE (LiDAR) [mm]")
g.add_legend(title="Plot ID")
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Final_products/Figures/{watershed}/{year}/{date}/"
    f"Plot_SWE_validation.png")
plt.close()