# This script validates the modelled snow density data to field data
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
resolution2 = var['resolution2'][0]
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
field=field[field['sample_type']=='Density']
field=field[field['density_percent'].notnull()]
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

Density_ids=[]
Density_eastings=[]
Density_northings=[]
FieldDensities=[]
for n in phases:
    x = field[(field['aco_survey']==n)]
    plot_id=np.array(x['plot_name'])
    plot_id=plot_id.reshape(len(plot_id),)
    easting=np.array(x['easting']).astype('float64')
    easting=easting.reshape(len(easting),)
    northing=np.array(x['northing']).astype('float64')
    northing=northing.reshape(len(northing),)
    density=np.array(x['density_percent']).astype('float64')
    density=density.reshape(len(density),)
    Density_ids.append(plot_id)
    Density_eastings.append(easting)
    Density_northings.append(northing)
    FieldDensities.append(density)
del x,n,plot_id,easting,northing,density

LidarDensities=[] # in g/cm3
for n in range(len(phases)):
    x=rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif')    
    LidarDensities.append(x)
del n,x

# Sample gridded datasets at field data collection points
Density_coord_list=[]
for m in range(len(phases)):
    coord_list=[]
    easting=Density_eastings[m]
    northing=Density_northings[m]
    for n in range(len(easting)):
        cl=(easting[n],northing[n])
        coord_list.append(cl)
    Density_coord_list.append(coord_list)
LidarDensities_sampled=[]
for n in range(len(phases)):
    x=[x for x in LidarDensities[n].sample(Density_coord_list[n])]
    x=np.array(x)
    LidarDensities_sampled.append(x)
del n,m,Density_coord_list,Density_eastings,easting,northing,Density_northings,cl,coord_list,x

##### Plot-averaged comparisons ---------------------------------------------------------------
FieldDensity_plot_mean=[]
FieldDensity_plot_sd=[]
FieldDensity_plot_names=[]
for n in range(len(phases)):
    fd=FieldDensities[n]
    fd=fd.reshape(len(fd),1)
    plot_id=Density_ids[n]
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
    FieldDensity_plot_mean.append(b)
    FieldDensity_plot_sd.append(a)
    FieldDensity_plot_names.append(output_list)

LidarDensity_plot_mean=[]
LidarDensity_plot_sd=[]
LidarDensity_plot_names=[]
for n in range(len(phases)):
    fd=LidarDensities_sampled[n]
    fd=fd.reshape(len(fd),1)
    plot_id=Density_ids[n]
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
    LidarDensity_plot_mean.append(b)
    LidarDensity_plot_sd.append(a)
    LidarDensity_plot_names.append(output_list)

#### Difference (LiDAR - Field) statistics----------------------------------------------------------------------------------
# Calculate field-LiDAR differences
Density_diffs=[]
Density_meandiff=[]
Density_sddiff=[]
Density_rmsediff=[]
for n in range(len(LidarDensities_sampled)):
    x=LidarDensities_sampled[n].reshape(len(LidarDensities_sampled[n],))
    y=(x-FieldDensities[n]) #both in g/cm3
    y[np.abs(y) > 0.3] =np.nan
    z=np.nanmean(y)
    zz=np.nanstd(y)
    mse=(np.nansum(y**2))/len(y)
    rmse=math.sqrt(mse)
    Density_diffs.append(y)
    Density_meandiff.append(z)
    Density_sddiff.append(zz)
    Density_rmsediff.append(rmse)
del n,x,y,z,zz

# Calculate mean differences for each individual plot
Density_plot_meandiffs=[]
Density_plot_sddiffs=[]
Density_plot_names=[]
for n in range(len(phases)):
    Dendiff=Density_diffs[n]
    Dendiff=Dendiff.reshape(len(Dendiff),1)
    plot_id=Density_ids[n]
    pp=[]
    for x in range(len(plot_id)):
        p=str(plot_id[x])
        pp.append(p)
    output_list = []
    for word in pp:
        if word not in output_list:
            output_list.append(word)
    Density_plot_meandiff=[]
    Density_plot_sddiff=[]
    pp=np.array(pp)
    for y in range(len(output_list)):
        i=np.where(pp==str(output_list[y]))
        ppp=Dendiff[i]
        mean=np.nanmean(ppp)
        sd=np.nanstd(ppp)
        Density_plot_meandiff.append(mean)
        Density_plot_sddiff.append(sd)
    Density_plot_meandiff=np.array(Density_plot_meandiff)
    Density_plot_meandiff=Density_plot_meandiff.reshape(len(Density_plot_meandiff),1)
    Density_plot_sddiff=np.array(Density_plot_sddiff)
    Density_plot_sddiff=Density_plot_sddiff.reshape(len(Density_plot_sddiff),1)
    Density_plot_meandiffs.append(Density_plot_meandiff)
    Density_plot_sddiffs.append(Density_plot_sddiff)
    Density_plot_names.append(output_list)

# Output ------------------------------------------------------------------------------------------------------------------
# Export difference statistics
# For the entire watershed
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/')
Density_field=pd.DataFrame(list(zip(phases,Density_meandiff,Density_sddiff,Density_rmsediff)),columns=['survey','Density_mean_diff_g*cm^-3','Density_sd_diff_g*cm^-3','Density_rmse_diff_g*cm^-3'])
Density_field.dropna(subset=['Density_mean_diff_g*cm^-3'], inplace=True)
Density_field.to_csv(str(extent)+'_field_validation_Density.csv', index=False)

# By plot
y = []
for n in range(len(phases)):
    surveys=np.repeat(phases[n],len(Density_plot_names[n]))
    x=pd.DataFrame(list(zip(surveys,Density_plot_names[n],FieldDensity_plot_mean[n],FieldDensity_plot_sd[n],LidarDensity_plot_mean[n],LidarDensity_plot_sd[n],Density_plot_meandiffs[n].flatten(),Density_plot_sddiffs[n].flatten())),columns=['survey','Plot_id','Field_density_mean','Field_density_sd','Lidar_density_mean','Lidar_density_sd','Density_mean_diff_g*cm^-3','Density_sd_diff_g*cm^-3'])
    y.append(x)
Density_plot=pd.concat(y)
Density_plot.to_csv(str(extent)+'_field_validation_by_plot_Density.csv', index=False)

Density_plot = Density_plot[Density_plot['Field_density_mean'] > 0]
maxvalue=np.round(np.max(Density_plot['Lidar_density_mean']) + 1,decimals = 1)
g = sns.FacetGrid(Density_plot, col='survey', hue='Plot_id', col_wrap=3)
def plot(x, y, xerr, yerr, **kwargs):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt = 'o', **kwargs)
    plt.grid(True)
g = g.map(plot, 'Field_density_mean', 'Lidar_density_mean', 'Field_density_sd', 'Lidar_density_sd')
def plot_one_to_one(x, y, **kwargs):
    ax = plt.gca()
    min_val = 0.1
    max_val = 0.6
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', linewidth=1)
g = g.map(plot_one_to_one, 'Field_density_mean', 'Lidar_density_mean')
for ax, name in zip(g.axes.flat, Density_field['survey']):
    meandiff = Density_field[Density_field['survey'] == name]['Density_mean_diff_g*cm^-3'].iloc[0]
    rmse = Density_field[Density_field['survey'] == name]['Density_rmse_diff_g*cm^-3'].iloc[0]
    text_label = f"Mean diff: {meandiff:.2f} g*cm^-3\nRMSE: {rmse:.2f} g*cm^-3"
    # Add text using ax.text(x_pos, y_pos, text)
    # The coordinates (x, y) are in data units for that specific subplot
    ax.text(0.30, maxvalue-1.2, text_label, fontsize=9, color='black', ha='left', va='center')
g.set_xlabels("Mean Density (Field) [g*cm^-3]")
g.set_ylabels("Mean Density (LiDAR) [g*cm^-3]")
g.add_legend(title="Plot ID")
for ax in g.axes.flat:
    ax.set_xlim(0.15, 0.55)
    ax.set_ylim(0.15, 0.55)
    ax.set_aspect('equal')
    ticks = np.arange(0.2, 0.6, 0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
    f"Plot_density_validation.png")
plt.close()