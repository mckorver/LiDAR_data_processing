# This script validates the snow depth LiDAR derived data to field data
# This script outputs:
# csv files of plot averaged statistics (mean, sd) and difference (LiDAR - field) statistics
# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

import rasterio
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Import input data ------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/input_data/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
phases = []
x = var['phases'][var['phases'].notna()]
for n in range(len(x)):
    a = x[n]
    phases.append(a)

## QAQC field data
datetimes_field=[]
datetimes_aco=[]
eastings=[]
northings=[]
depth_ids=[]
depths=[]
cores=[]
densities=[]
manual_remove=[]
field_phases=[]
os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/'+str(year))
for b in range(len(phases)):
    # Read field data collection coordinates
    datetime_f=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['plot_datetime'],parse_dates=['plot_datetime']))
    datetime_f=datetime_f.reshape(len(datetime_f),)
    datetime_l=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['aco_datetime'],parse_dates=['aco_datetime']))
    datetime_l=datetime_l.reshape(len(datetime_l),)
    easting=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['easting_m'])).astype('float64')
    easting=easting.reshape(len(easting),)
    northing=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['northing_m'])).astype('float64')
    northing=northing.reshape(len(northing),)
    plot_id=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['plot_id']))
    plot_id=plot_id.reshape(len(plot_id),)
    depth=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['snow_depth'])).astype('float64')
    depth=depth.reshape(len(depth),)
    core=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['core_length_final'])).astype('float64')
    core=core.reshape(len(core),)
    density=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['density'])).astype('float64')
    density=density.reshape(len(density),)
    manual_rem=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[b])+'.csv', usecols=['manual_remove']))
    manual_rem=manual_rem.reshape(len(manual_rem),)
    field_phase=np.array(phases[b])
    field_phase=np.repeat(field_phase,len(easting))
    datetimes_field.append(datetime_f)
    datetimes_aco.append(datetime_l)
    eastings.append(easting)
    northings.append(northing)
    depth_ids.append(plot_id)
    depths.append(depth)
    cores.append(core)
    densities.append(density)
    manual_remove.append(manual_rem)
    field_phases.append(field_phase)

df=pd.DataFrame({"phase":np.concatenate(field_phases),"datetime_field":np.concatenate(datetimes_field),"datetime_aco":np.concatenate(datetimes_aco),
                 "easting_m":np.concatenate(eastings),"northing_m":np.concatenate(northings),"plot_id":np.concatenate(depth_ids),
                 "snow_depth":np.concatenate(depths),"core_depth":np.concatenate(cores),
                 "density":np.concatenate(densities), "manual_remove":np.concatenate(manual_remove)})
df['time_gap_hr']=df['datetime_field']-df['datetime_aco']
df['time_gap_hr'] = df['time_gap_hr'].dt.total_seconds()/3600
df['time_gap_hr'] = df['time_gap_hr'].abs().astype('int64')
df['year'] = df['datetime_field'].dt.year.astype('string')
df['month'] = df['datetime_field'].dt.month
df['snow_depth_m'] = df['snow_depth']/100

# QAQC field data
df=df[(df['snow_depth'].notna())]
df['flag'] = 'AV'
df.loc[(df['time_gap_hr']>60), 'flag'] = 'time'
df.loc[(df['manual_remove']=='Y'), 'flag'] = 'manual'
df.loc[(df['snow_depth']-df['core_depth']<-5)|(df['snow_depth']/df['core_depth']>=2), 'flag'] = 'core'
filt=df[(df['flag']=='AV')]

Depth_ids=[]
Depth_eastings=[]
Depth_northings=[]
FieldDepths=[]
for n in phases:
    x = filt[(filt['phase']==n)]
    y = x[x['snow_depth'].notnull()]
    plot_id=np.array(y['plot_id'])
    plot_id=plot_id.reshape(len(plot_id),)
    easting=np.array(y['easting_m']).astype('float64')
    easting=easting.reshape(len(easting),)
    northing=np.array(y['northing_m']).astype('float64')
    northing=northing.reshape(len(northing),)
    depth=np.array(y['snow_depth']).astype('float64')
    depth=depth.reshape(len(depth),)
    Depth_ids.append(plot_id)
    Depth_eastings.append(easting)
    Depth_northings.append(northing)
    FieldDepths.append(depth)
del x,y,n,plot_id,easting,northing,depth
                      
# Read gridded input datasets
LidarDepths=[] #in m
for n in range(len(phases)):
    x=rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/Provisional/resolution_'+str(resolution1)+'m/Provisional_SD_'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_capped_clipped_vegcorrected_'+str(resolution1)+'m.tif')    
    LidarDepths.append(x)
del n,x

# Sample gridded datasets at field data collection points
Depth_coord_list=[]
for m in range(len(phases)):
    coord_list=[]
    easting=Depth_eastings[m]
    northing=Depth_northings[m]
    for n in range(len(easting)):
        cl=(easting[n],northing[n])
        coord_list.append(cl)
    Depth_coord_list.append(coord_list)
LidarDepths_sampled=[]
for n in range(len(phases)):
    x=[x for x in LidarDepths[n].sample(Depth_coord_list[n])]
    x=np.array(x)
    i=np.where(x<0) # Optional, define which LiDAR derived snowdepth values to exclude
    x[i]=np.nan
    LidarDepths_sampled.append(x)
del n,m,Depth_coord_list,Depth_eastings,easting,northing,Depth_northings,cl,coord_list,x

##### Plot-averaged comparisons ---------------------------------------------------------------
FieldDepth_plot_mean=[]
FieldDepth_plot_sd=[]
FieldDepth_plot_names=[]
for n in range(len(phases)):
    fd=FieldDepths[n]/100 #convert to m
    fd=fd.reshape(len(fd),1)
    plot_id=Depth_ids[n]
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
        mean=np.nanmean(ppp)
        sd=np.nanstd(ppp)
        b.append(mean)
        a.append(sd)
    FieldDepth_plot_mean.append(b)
    FieldDepth_plot_sd.append(a)
    FieldDepth_plot_names.append(output_list)

LidarDepth_plot_mean=[]
LidarDepth_plot_sd=[]
LidarDepth_plot_names=[]
for n in range(len(phases)):
    fd=LidarDepths_sampled[n]
    fd=fd.reshape(len(fd),1)
    plot_id=Depth_ids[n]
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
    LidarDepth_plot_mean.append(b)
    LidarDepth_plot_sd.append(a)
    LidarDepth_plot_names.append(output_list)

#### Difference (LiDAR - Field) statistics----------------------------------------------------------------------------------
# Calculate field-LiDAR differences
Depth_diffs=[]
Depth_meandiff=[]
Depth_sddiff=[]
Depth_rmsediff=[]
for n in range(len(LidarDepths_sampled)):
    x=LidarDepths_sampled[n].reshape(len(LidarDepths_sampled[n],))
    y=(x-(FieldDepths[n]/100)) #convert FieldDepths in cm to m
    z=np.nanmean(y)
    zz=np.nanstd(y)
    mse=(np.nansum(y**2))/len(y)
    rmse=math.sqrt(mse)
    Depth_diffs.append(y)
    Depth_meandiff.append(z)
    Depth_sddiff.append(zz)
    Depth_rmsediff.append(rmse)
del n,x,y,z,zz

# Calculate mean differences for each individual plot
Depth_plot_meandiffs=[]
Depth_plot_sddiffs=[]
Depth_plot_names=[]
for n in range(len(phases)):
    Depdiff=Depth_diffs[n]
    Depdiff=Depdiff.reshape(len(Depdiff),1)
    plot_id=Depth_ids[n]
    pp=[]
    for x in range(len(plot_id)):
        p=str(plot_id[x])
        pp.append(p)
    output_list = []
    for word in pp:
        if word not in output_list:
            output_list.append(word)
    Depth_plot_meandiff=[]
    Depth_plot_sddiff=[]
    pp=np.array(pp)
    for y in range(len(output_list)):
        i=np.where(pp==str(output_list[y]))
        ppp=Depdiff[i]
        mean=np.nanmean(ppp)
        sd=np.nanstd(ppp)
        Depth_plot_meandiff.append(mean)
        Depth_plot_sddiff.append(sd)
    Depth_plot_meandiff=np.array(Depth_plot_meandiff)
    Depth_plot_meandiff=Depth_plot_meandiff.reshape(len(Depth_plot_meandiff),1)
    Depth_plot_sddiff=np.array(Depth_plot_sddiff)
    Depth_plot_sddiff=Depth_plot_sddiff.reshape(len(Depth_plot_sddiff),1)
    Depth_plot_meandiffs.append(Depth_plot_meandiff)
    Depth_plot_sddiffs.append(Depth_plot_sddiff)
    Depth_plot_names.append(output_list)

# Output ------------------------------------------------------------------------------------------------------------------
# Export difference statistics
# For the entire watershed
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/')
Depth_field=pd.DataFrame(list(zip(phases,Depth_meandiff,Depth_sddiff,Depth_rmsediff)),columns=['survey','Depth_mean_diff_m','Depth_sd_diff_m','Depth_rmse_diff_m'])
Depth_field.to_csv(str(extent)+'_field_validation_Depth.csv', index=False)

# By plot
y = []
for n in range(len(phases)):
    surveys=np.repeat(phases[n],len(Depth_plot_names[n]))
    x=pd.DataFrame(list(zip(surveys,Depth_plot_names[n],FieldDepth_plot_mean[n],FieldDepth_plot_sd[n],LidarDepth_plot_mean[n],LidarDepth_plot_sd[n],Depth_plot_meandiffs[n].flatten(),Depth_plot_sddiffs[n].flatten())),columns=['survey','Plot_id','Field_Depth_mean','Field_Depth_sd','Lidar_Depth_mean','Lidar_Depth_sd','Depth_mean_diff_m','Depth_sd_diff_m'])
    y.append(x)
Depth_plot = pd.concat(y)
Depth_plot.to_csv(str(extent)+'_field_validation_by_plot_Depth.csv', index=False)
    
maxvalue=np.round(np.max(Depth_plot['Lidar_Depth_mean']) + 1,decimals = 1)
g = sns.FacetGrid(Depth_plot, col='survey',hue='Plot_id')
def plot(x, y, xerr, yerr, **kwargs):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt = 'o', **kwargs)
    plt.grid(True)
g = g.map(plot, 'Field_Depth_mean', 'Lidar_Depth_mean', 'Field_Depth_sd', 'Lidar_Depth_sd')
def plot_one_to_one(**kwargs):
    ax = plt.gca()
    min_val = 0
    max_val = maxvalue
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=1,**kwargs)
g = g.map(plot_one_to_one, color = 'darkgrey')
for ax, name in zip(g.axes.flat, Depth_field['survey']):
    value = Depth_field[Depth_field['survey'] == name]['Depth_mean_diff_m'].iloc[0]
    text_label = f"Mean diff:\n{value:.2f} m"
    # Add text using ax.text(x_pos, y_pos, text)
    # The coordinates (x, y) are in data units for that specific subplot
    ax.text(0.5, maxvalue-1, text_label, fontsize=9, color='black', ha='left', va='center')
g.set_xlabels("Mean Depth (Field) [m]")
g.set_ylabels("Mean Depth (LiDAR) [m]")
g.set(xlim=(0,maxvalue), ylim=(0,maxvalue),xticks=np.arange(0,maxvalue,1),yticks=np.arange(0,maxvalue,1))
g.add_legend(title="Plot ID")
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
    f"Plot_depth_validation.png")
plt.close()