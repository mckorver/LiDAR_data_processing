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

# Import input data ------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/input_data/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution2 = var['resolution2'][0]
lakemodel = var['lakemodel'][0]
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
df=df[(df['density'].notna())&(df['density']>0)]
df['flag'] = 'AV'
df.loc[(df['time_gap_hr']>60), 'flag'] = 'time'
df.loc[(df['manual_remove']=='Y'), 'flag'] = 'manual'
df.loc[(df['density']>0.8)|(df['density']<0.1), 'flag'] = 'range'
df.loc[(df['snow_depth']-df['core_depth']<-5)|(df['snow_depth']/df['core_depth']>=2), 'flag'] = 'core'
filt=df[(df['flag']=='AV')]

Density_ids=[]
Density_eastings=[]
Density_northings=[]
FieldDensities=[]
for n in phases:
    x = filt[(filt['phase']==n)]
    y = x[x['density'].notnull()]
    plot_id=np.array(y['plot_id'])
    plot_id=plot_id.reshape(len(plot_id),)
    easting=np.array(y['easting_m']).astype('float64')
    easting=easting.reshape(len(easting),)
    northing=np.array(y['northing_m']).astype('float64')
    northing=northing.reshape(len(northing),)
    density=np.array(y['density']).astype('float64')
    density=density.reshape(len(density),)
    Density_ids.append(plot_id)
    Density_eastings.append(easting)
    Density_northings.append(northing)
    FieldDensities.append(density)
del x,y,n,plot_id,easting,northing,density

LidarDensities=[] # in g/cm3
for n in range(len(phases)):
    x=rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'.tif')    
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
g = sns.FacetGrid(Density_plot, col='survey',hue='Plot_id')
def plot(x, y, xerr, yerr, **kwargs):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt = 'o', **kwargs)
    plt.grid(True)
g = g.map(plot, 'Field_density_mean', 'Lidar_density_mean', 'Field_density_sd', 'Lidar_density_sd')
def plot_one_to_one(x, y, **kwargs):
    ax = plt.gca()
    min_val = 0.3
    max_val = 0.5
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--',**kwargs)
g = g.map(plot_one_to_one, 'Field_density_mean', 'Lidar_density_mean')
for ax, name in zip(g.axes.flat, Density_field['survey']):
    value = Density_field[Density_field['survey'] == name]['Density_mean_diff_g*cm^-3'].iloc[0]
    text_label = f"Mean diff:\n{value:.2f} g/cm3"
    # Add text using ax.text(x_pos, y_pos, text)
    # The coordinates (x, y) are in data units for that specific subplot
    ax.text(0.33, maxvalue-0.95, text_label, fontsize=9, color='black', ha='left', va='center')
g.set_xlabels("Mean Density (Field) [g*cm^-3]")
g.set_ylabels("Mean Density (LiDAR) [g*cm^-3]")
g.add_legend(title="Plot ID")
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
    f"Plot_density_validation.png")
plt.close()