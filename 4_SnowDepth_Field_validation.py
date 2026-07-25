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
# Survey naming convention
if lidar.lower() == "aco":
    survey = "aco_survey"
elif lidar.lower() == "rpas":
    survey = "rpas_survey"
else:
    raise ValueError(f"Unknown lidar type: {lidar}")
resolution1 = var['resolution1'][0]
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
field=field[field['sample_type']=='Depth']
field=field[field['probe_depth_cm'].notnull()]
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
field.loc[(field['probe_depth_cm']<0)|(field['probe_depth_cm']>1000), 'flag2'] = 'out of range'
field.loc[(field['sample_rating'].notna()) & (field['sample_rating']<4), 'flag3'] = 'low sample rating'
field.loc[(field['time_gap_hr']>60), 'flag4'] = 'time mismatch'
field=field[(field['flag2']=='AV')&(field['flag3']=='AV')&(field['flag4']=='AV')] # Select which flags you want to apply here
for n in range(len(phases)):
    field = field.drop(columns='survey_timestamp'+str(n+1))
field = field.drop(columns=['retrieval_percent','sample_rating','density_frac','swe_cm','core_depth_cm','sample_type','basin_name','region_name','coord_water_year','closest_id','time_gap_hr','flag1','flag2','flag3','flag4','latitude','longitude'])

# Group and average snow depths by plot_id
field_gr=field.groupby(['plot_name', 'datetime', survey, 'easting', 'northing'])['probe_depth_cm'].agg(['mean', 'std']).reset_index()
field_gr=field_gr.rename(columns={"mean": "Field_Depth_mean_m", "std": "Field_Depth_sd_m"})
field_gr['Field_Depth_mean_m']=field_gr['Field_Depth_mean_m']/100
field_gr['Field_Depth_sd_m']=field_gr['Field_Depth_sd_m']/100

# Sample LiDAR derived snow depth for field locations
LiDAR_mean=[]
LiDAR_sd=[]
for n in range(len(phases)):
    gr = field_gr[field_gr[survey] == phases[n]].reset_index(drop=True)
    buffer = 10 # buffer distance around centre point of cardinal plot
    src = rasterio.open(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Maps/SnowDepth/resolution_'+str(resolution1)+'m/'+str(extent)+'_'+str(year)+'_'+str(phases[n])+'_SnowDepth_lakemodel'+str(lakemodel)+'_'+str(resolution1)+'m.tif')
    a=[]
    b=[]
    for x in range(len(gr)):
        coords = (gr.at[x,'easting'], gr.at[x,'northing'])
        point = Point(coords)
        buffer_poly = point.buffer(buffer)
        # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
        mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src.shape,transform=src.transform,invert=False)
        data = src.read(1).astype(float) # Read data and apply mask. Read first band
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        sampled_values = data[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
        sampled_mean = np.nanmean(sampled_values) # Take the mean of all pixels within buffer
        sampled_sd = np.nanstd(sampled_values) # Take the std of all pixels within buffer
        a.append(sampled_mean)
        b.append(sampled_sd)
    LiDAR_mean.append(a)
    LiDAR_sd.append(b)

# Add LiDAR Depth metrics to field dataframe
df_cols=field_gr.columns.tolist()
df_cols.append('LiDAR_Depth_mean_m')
df_cols.append('LiDAR_Depth_sd_m')
merged=pd.DataFrame(columns=df_cols)
for n in range(len(phases)):
    x = field_gr[field_gr[survey] == phases[n]].copy()
    x['LiDAR_Depth_mean_m'] = LiDAR_mean[n]
    x['LiDAR_Depth_sd_m'] = LiDAR_sd[n]
    merged=pd.concat([merged,x],ignore_index=True)
merged['Depth_mean_diff_m']=merged['LiDAR_Depth_mean_m']-merged['Field_Depth_mean_m']
merged['Depth_sd_diff_m']=merged['LiDAR_Depth_sd_m']-merged['Field_Depth_sd_m']

# Calculate mean, sd, and RMSE of differences (plot means) for each survey
#survey_stats=pd.DataFrame(columns=['aco_survey','Depth_mean_diff_m','Depth_sd_diff_m','Depth_rmse_diff_m'])
survey_stats = pd.DataFrame(columns=[survey, 'Depth_mean_diff_m', 'Depth_sd_diff_m', 'Depth_rmse_diff_m'])
for phase in phases:
    x = merged[merged[survey] == phase]
    if len(x) == 0:
        y = pd.DataFrame([{survey: phase, 'Depth_mean_diff_m': np.nan, 'Depth_sd_diff_m': np.nan, 'Depth_rmse_diff_m': np.nan}])
    else:
        a = x['Depth_mean_diff_m'].mean()
        b = x['Depth_mean_diff_m'].std()
        c = math.sqrt(np.nansum(x['Depth_mean_diff_m']**2) / len(x))
        y = pd.DataFrame([{survey: phase, 'Depth_mean_diff_m': a, 'Depth_sd_diff_m': b, 'Depth_rmse_diff_m': c}])
    survey_stats = pd.concat([survey_stats, y], ignore_index=True)

# Output ------------------------------------------------------------------------------------------------------------------
# Export difference statistics
# For the entire watershed
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Bias_analysis/'+str(watershed)+'/'+str(year)+'/')
survey_stats.to_csv(str(extent)+'_field_validation_Depth.csv', index=False)
merged.to_csv(str(extent)+'_field_validation_by_plot_Depth.csv', index=False)

maxvalue=np.round(np.max(merged['LiDAR_Depth_mean_m']) + 1,decimals = 1)
g = sns.FacetGrid(merged, col=survey, hue='plot_name', aspect=1, col_wrap=3)
def plot(x, y, xerr, yerr, **kwargs):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt = 'o', **kwargs)
    plt.grid(True)
g = g.map(plot, 'Field_Depth_mean_m', 'LiDAR_Depth_mean_m', 'Field_Depth_sd_m', 'LiDAR_Depth_sd_m')
def plot_one_to_one(**kwargs):
    ax = plt.gca()
    min_val = 0
    max_val = maxvalue
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=1,**kwargs)
g = g.map(plot_one_to_one, color = 'darkgrey')
#for ax, name in zip(g.axes.flat, survey_stats['aco_survey']):
for ax, name in zip(g.axes.flat, survey_stats[survey]):
    meandiff = survey_stats[survey_stats[survey] == name]['Depth_mean_diff_m'].iloc[0]
    rmse = survey_stats[survey_stats[survey] == name]['Depth_rmse_diff_m'].iloc[0]
    text_label = f"Mean diff: {meandiff:.2f} m\nRMSE: {rmse:.2f} m"
    # Add text using ax.text(x_pos, y_pos, text)
    # The coordinates (x, y) are in data units for that specific subplot
    ax.text(0.25, maxvalue-0.5, text_label, fontsize=9, color='black', ha='left', va='center')
g.set_xlabels("Mean Depth (Field) [m]")
g.set_ylabels("Mean Depth (LiDAR) [m]")
g.set(xlim=(0,maxvalue), ylim=(0,maxvalue),xticks=np.arange(0,maxvalue,1),yticks=np.arange(0,maxvalue,1))
g.add_legend(title="Plot ID")
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Bias_analysis/{watershed}/{year}/"
    f"Plot_depth_validation.png")
plt.close()