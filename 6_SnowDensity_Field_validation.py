# This script validates the modelled snow density data to field data
# This script outputs:
# csv files of plot averaged statistics (mean, sd) and difference (LiDAR - field) statistics
# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

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
from shapely.geometry import Point
import rasterio.features
import rasterio

# Import input data ------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('E:/LiDAR_data_processing/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
if lidar.lower() == "aco":
    survey = "aco_survey"
    survey_prefix = "P"
else:
    survey = "rpas_survey"
    survey_prefix = "D"
resolution2 = var['resolution2'][0]
lakemodel = var['lakemodel'][0]
phases = []
dates=[]
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if isinstance(y, float):
            a.append(int(y))
        else:
            a.append(y)
append_fun(phases,'phases')
append_fun(dates,'dates')
if watershed == 'TSI':
    region_name = 'TSITIKA' 
elif watershed == 'CRU':
    region_name = 'CRUICKSHANK'
elif watershed == 'ENG':
    region_name = 'ENGLISHMAN'
elif watershed == 'MV':
    region_name = 'METRO VANCOUVER'

# Calculate 'day in season'
#survey_dates=[datetime.strptime(date, "%Y-%m-%d") for date in dates]
survey_dates = pd.to_datetime(dates).tolist()
months=[d.month for d in survey_dates]
years=[d.year for d in survey_dates]
day_in_season=[]
for n in range(len(survey_dates)):
    if months[n] <9:
        x=datetime.strptime(str(years[n]-1)+"-09-01", "%Y-%m-%d")
    else:
        x=datetime.strptime(str(years[n])+"-09-01", "%Y-%m-%d")
    y=survey_dates[n]-x
    y=y.total_seconds()/86400
    day_in_season.append(y)

## Import, edit, and filter field snowplot data -------------------------------------
os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/')
field=pd.read_csv('snowplot.csv')
field['datetime'] = field['datetime'].astype('datetime64[ns]')
field=field[field['region_name']==region_name]
field['water_year'] = field['datetime'].dt.year
field=field[field['water_year']==int(year)]
field=field[field['sample_type']=='Density']
field=field[field['density_frac'].notnull()]
easting, northing, zone_number, zone_letter = utm.from_latlon(
    field['latitude'].values, 
    field['longitude'].values)
field['easting'] = easting
field['northing'] = northing

# Add a column indicating ACO or drone flight datetime and calculate time between field surveys and ACO flights
cols=[]
for n in range(len(phases)):
    field['survey_timestamp'+str(n+1)] = survey_dates[n] + timedelta(hours=24)
    cols.append('survey_timestamp'+str(n+1))
abs_diff = field[cols].sub(field['datetime'], axis=0).abs()
field['closest_id'] = abs_diff.idxmin(axis=1)
#field['survey_timestamp'] = field.lookup(field.index, field['closest_id']) if hasattr(field, 'lookup') else field.apply(lambda row: row[row['closest_id']], axis=1)
cols_idx = field.columns.get_indexer(field["closest_id"])
rows_idx = np.arange(len(field))
field["survey_timestamp"] = field.to_numpy()[rows_idx, cols_idx]
lst=[]
for n in range(len(phases)):
    x = field.loc[(field['survey_timestamp']==(survey_dates[n] + timedelta(hours=24)))]
    x = field.loc[field['survey_timestamp'] == (survey_dates[n] + timedelta(hours=24))].copy()
    x[survey] = phases[n]
    lst.append(x)
field = pd.concat(lst, ignore_index=True)
field['time_gap_hr']=field['datetime']-field['survey_timestamp']
field['time_gap_hr'] = field['time_gap_hr'].dt.total_seconds()/3600
field['time_gap_hr'] = field['time_gap_hr'].abs()

# QAQC field data
field[['flag1','flag2','flag3','flag4']] = 'AV'
field.loc[(field['water_year']!=field['coord_water_year']), 'flag1'] = 'coord uncertainty'
field.loc[(field['density_frac']<0.1)|(field['density_frac']>0.65), 'flag2'] = 'out of range'
field.loc[(field['sample_rating'].notna()) & (field['sample_rating']<4), 'flag3'] = 'low sample rating'
field.loc[(field['time_gap_hr']>60), 'flag4'] = 'time mismatch'
field=field[(field['flag2']=='AV')&(field['flag3']=='AV')&(field['flag4']=='AV')] # Select which flags you want to apply here
for n in range(len(phases)):
    field = field.drop(columns='survey_timestamp'+str(n+1))
field = field.drop(columns=['retrieval_percent','sample_rating','probe_depth_cm','swe_cm','sample_type','basin_name','region_name','coord_water_year','closest_id','time_gap_hr','flag1','flag2','flag3','flag4','latitude','longitude'])

# Group and average snow densities by plot_id
field_gr=field.groupby(['plot_name','datetime',survey,'easting','northing'])['density_frac'].agg(['mean','std']).reset_index()
field_gr=field_gr.rename(columns={"mean": "Field_Density_mean_gcm3", "std": "Field_Density_sd_gcm3"})

# Sample LiDAR derived snow densities for field locations
LiDAR_mean=[]
LiDAR_sd=[]
for n in range(len(phases)):
    gr=field_gr[field_gr[survey] == survey_prefix + str(n+1)].reset_index(drop=True)
    buffer = 10 # buffer distance around centre point of cardinal plot
    src = rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDensity/resolution_'+str(resolution2)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDensity_lakemodel'+str(lakemodel)+'_'+str(resolution2)+'m.tif')
    a=[]
    b=[]
    for x in range(len(gr)):
        coords = (gr.at[x,'easting'], gr.at[x,'northing'])
        point = Point(coords)
        buffer_poly = point.buffer(buffer)
        # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
        mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src.shape,transform=src.transform,invert=False)
        data = src.read(1) # Read data and apply mask. Read first band
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        sampled_values = data[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
        sampled_mean = np.nanmean(sampled_values) # Take the mean of all pixels within buffer
        sampled_sd = np.nanstd(sampled_values) # Take the std of all pixels within buffer
        a.append(sampled_mean)
        b.append(sampled_sd)
    LiDAR_mean.append(a)
    LiDAR_sd.append(b)

# Add LiDAR Density metrics to field dataframe
df_cols=field_gr.columns.tolist()
df_cols.append('LiDAR_Density_mean_gcm3')
df_cols.append('LiDAR_Density_sd_gcm3')
merged=pd.DataFrame(columns=df_cols)
for n in range(len(phases)):
    x = field_gr[field_gr[survey] == survey_prefix + str(n+1)].copy()
    x['LiDAR_Density_mean_gcm3'] = LiDAR_mean[n]
    x['LiDAR_Density_sd_gcm3'] = LiDAR_sd[n]
    merged=pd.concat([merged,x],ignore_index=True)
merged['Density_mean_diff_gcm3']=merged['LiDAR_Density_mean_gcm3']-merged['Field_Density_mean_gcm3']
merged['Density_sd_diff_gcm3']=merged['LiDAR_Density_sd_gcm3']-merged['Field_Density_sd_gcm3']

# Calculate mean, sd, and RMSE of differences (plot means) for each survey
#survey_stats=pd.DataFrame(columns=['aco_survey','Density_mean_diff_gcm3','Density_sd_diff_gcm3','Density_rmse_diff_gcm3'])
survey_stats = pd.DataFrame(columns=[survey, 'Density_mean_diff_gcm3', 'Density_sd_diff_gcm3', 'Density_rmse_diff_gcm3'])
for n in range(len(phases)):
    x = merged[merged[survey] == survey_prefix + str(n+1)]
    if len(x)==0:
       y=pd.DataFrame([{survey: survey_prefix + str(n+1),'Density_mean_diff_gcm3':np.nan,'Density_sd_diff_gcm3':np.nan,'Density_rmse_diff_gcm3':np.nan}]) 
    else:
        a=x['Density_mean_diff_gcm3'].mean()
        b=x['Density_mean_diff_gcm3'].std()
        c=math.sqrt((np.nansum(x['Density_mean_diff_gcm3']**2))/len(x['Density_mean_diff_gcm3']))
        #y=pd.DataFrame([{'aco_survey':'P'+str(n+1),'Density_mean_diff_gcm3':a,'Density_sd_diff_gcm3':b,'Density_rmse_diff_gcm3':c}])
        y=pd.DataFrame([{survey: survey_prefix + str(n+1),'Density_mean_diff_gcm3':a,'Density_sd_diff_gcm3':b,'Density_rmse_diff_gcm3':c}])
    survey_stats=pd.concat([survey_stats,y],ignore_index=True)

# Output ------------------------------------------------------------------------------------------------------------------
# Export difference statistics
# For the entire watershed
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/')
survey_stats.to_csv(str(extent)+'_field_validation_Density.csv', index=False)
merged.to_csv(str(extent)+'_field_validation_by_plot_Density.csv', index=False)

#maxvalue=np.round(np.max(merged['LiDAR_Density_mean_gcm3']) + 0.1,decimals = 2)
#minvalue=np.round(np.min(merged['LiDAR_Density_mean_gcm3']) - 0.1,decimals = 2)
step = 0.05
maxvalue = step * np.ceil((merged['Field_Density_mean_gcm3'].max() + 0.02) / step)
minvalue = step * np.floor((merged['Field_Density_mean_gcm3'].min() - 0.02) / step)
g = sns.FacetGrid(merged, col=survey, hue='plot_name',col_wrap=3)
def plot(x, y, xerr, yerr, **kwargs):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt = 'o', **kwargs)
    plt.grid(True)
g = g.map(plot, 'Field_Density_mean_gcm3', 'LiDAR_Density_mean_gcm3', 'Field_Density_sd_gcm3', 'LiDAR_Density_sd_gcm3')
def plot_one_to_one(**kwargs):
    ax = plt.gca()
    min_val = minvalue
    max_val = maxvalue
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=1,**kwargs)
g = g.map(plot_one_to_one, color = 'darkgrey')
for ax, name in zip(g.axes.flat, survey_stats[survey]):
    meandiff = survey_stats[survey_stats[survey] == name]['Density_mean_diff_gcm3'].iloc[0]
    rmse = survey_stats[survey_stats[survey] == name]['Density_rmse_diff_gcm3'].iloc[0]
    text_label = f"Mean diff: {meandiff:.2f} g/cm3\nRMSE: {rmse:.2f} g/cm3"
    # Add text using ax.text(x_pos, y_pos, text)
    # The coordinates (x, y) are in data units for that specific subplot
    ax.text(minvalue+0.025, maxvalue-0.075, text_label, fontsize=9, color='black', ha='left', va='center')
    ax.set_aspect('equal', adjustable='box')
g.set_xlabels("Mean Density (Field) [g/cm3]")
g.set_ylabels("Mean Density (LiDAR) [g/cm3]")
g.set(xlim=(minvalue,maxvalue), ylim=(minvalue,maxvalue),xticks=np.arange(minvalue,maxvalue,0.05),yticks=np.arange(minvalue,maxvalue,0.05))
g.add_legend(title="Plot ID")
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=30)
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
    f"Plot_density_validation.png")
plt.close()