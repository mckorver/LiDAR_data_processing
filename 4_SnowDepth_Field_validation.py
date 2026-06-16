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
from datetime import datetime,timedelta
import utm

# Import input data ------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
lakemodel = var['lakemodel'][0]
phases = []
days_in_season=[]
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if b == 'days_in_season':
            a.append(int(y))
        else:
            a.append(y)
append_fun(phases,'phases')
append_fun(days_in_season,'days_in_season')
timestamp = []
for n in range(len(phases)):
    x = datetime.strptime(year + "-03-01 12:00:00", "%Y-%m-%d %H:%M:%S")
    y = x + timedelta(days=days_in_season[n])
    timestamp.append(str(y))

if watershed == 'TSI':
    region_name = 'Tsitika' 
elif watershed == 'CRU':
    region_name = 'Cruickshank'
elif watershed == 'ENG':
    region_name = 'Englishman'
elif watershed == 'MV':
    region_name = 'Metro Vancouver'

## Import, edit, and filter field snowplot data -------------------------------------
os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/')
field=pd.read_csv('snowplots.csv')
field=field[field['region_name']==region_name]
field=field[field['coord_year']==int(year)]
field=field[field['sample_type']=='Depth']
field=field[field['probe_depth_cm'].notnull()]
field['sample_timestamp'] = field['sample_timestamp'].astype('datetime64[ns]')
easting, northing, zone_number, zone_letter = utm.from_latlon(
    field['latitude'].values, 
    field['longitude'].values)
field['easting'] = easting
field['northing'] = northing
# Add a column indicating ACO or drone flight datetime and filter measurements taken >60 hrs later or earlier
candidate_cols=[]
for n in range(len(phases)):
    field['aco_timestamp'+str(n+1)] = timestamp[n]
    field['aco_timestamp'+str(n+1)] = field['aco_timestamp'+str(n+1)].astype('datetime64[ns]')
    candidate_cols.append('aco_timestamp'+str(n+1))
abs_diff = field[candidate_cols].sub(field['sample_timestamp'], axis=0).abs()
field['closest_id'] = abs_diff.idxmin(axis=1)
field['aco_timestamp'] = field.lookup(field.index, field['closest_id']) if hasattr(field, 'lookup') else field.apply(lambda row: row[row['closest_id']], axis=1)
lst=[]
for n in range(len(phases)):
    x = field.loc[(field['aco_timestamp']==timestamp[n])]
    x['aco_survey'] = phases[n]
    lst.append(x)
field = pd.concat(lst, ignore_index=True)
field['time_gap_hr']=field['sample_timestamp']-field['aco_timestamp']
field['time_gap_hr'] = field['time_gap_hr'].dt.total_seconds()/3600
field['time_gap_hr'] = field['time_gap_hr'].abs()
field['flag'] = 'AV'
field.loc[(field['time_gap_hr']>60), 'flag'] = 'time'
field=field.loc[(field['flag']=='AV')]
for n in range(len(phases)):
    field = field.drop(columns='aco_timestamp'+str(n+1))
field = field.drop(columns=['closest_id','time_gap_hr'])

Depth_ids=[]
Depth_eastings=[]
Depth_northings=[]
FieldDepths=[]
for n in phases:
    x = field[(field['aco_survey']==n)]
    plot_id=np.array(x['plot_name'])
    plot_id=plot_id.reshape(len(plot_id),)
    easting=np.array(x['easting']).astype('float64')
    easting=easting.reshape(len(easting),)
    northing=np.array(x['northing']).astype('float64')
    northing=northing.reshape(len(northing),)
    depth=np.array(x['probe_depth_cm']).astype('float64')
    depth=depth.reshape(len(depth),)
    Depth_ids.append(plot_id)
    Depth_eastings.append(easting)
    Depth_northings.append(northing)
    FieldDepths.append(depth)
del x,n,plot_id,easting,northing,depth

# Read gridded input datasets
LidarDepths=[] #in m
for n in range(len(phases)):
    x=rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_'+str(resolution1)+'m.tif')    
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
Depth_field.dropna(subset=['Depth_mean_diff_m'], inplace=True)
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
    meandiff = Depth_field[Depth_field['survey'] == name]['Depth_mean_diff_m'].iloc[0]
    rmse = Depth_field[Depth_field['survey'] == name]['Depth_rmse_diff_m'].iloc[0]
    text_label = f"Mean diff: {meandiff:.2f} m\nRMSE: {rmse:.2f} m"
    # Add text using ax.text(x_pos, y_pos, text)
    # The coordinates (x, y) are in data units for that specific subplot
    ax.text(0.3, maxvalue-1.2, text_label, fontsize=9, color='black', ha='left', va='center')
g.set_xlabels("Mean Depth (Field) [m]")
g.set_ylabels("Mean Depth (LiDAR) [m]")
g.set(xlim=(0,maxvalue), ylim=(0,maxvalue),xticks=np.arange(0,maxvalue,1),yticks=np.arange(0,maxvalue,1))
g.add_legend(title="Plot ID")
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
    f"Plot_depth_validation.png")
plt.close()