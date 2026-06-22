
# Import packages
import rasterio
import rasterio.features
from shapely.geometry import Point
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree
import joblib
import utm
import pyrsgis

# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/Density_modelling/ML_model_processing_variables.csv', dtype={'years':str, 'resolution1':str, 'BEversion':str, 'CANversion':str, 'DENSversion':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
BEversion = var['BEversion'][0]
CANversion = var['CANversion'][0]
DENSversion = var['DENSversion'][0]
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if isinstance(y, float):
            a.append(int(y))
        else:
            a.append(y)
years=[]
predictors = []
phase=[]
append_fun(years,'years')
append_fun(predictors,'predictors')
append_fun(phase,'phases')
phases=[]
for n in range(len(phase)):
    a=[]
    for m in range(phase[n]):
        x='P'+str(m+1)
        a.append(x)
    phases.append(a)
if watershed == 'TSI':
    region_name = 'Tsitika' 
elif watershed == 'CRU':
    region_name = 'Cruickshank'
elif watershed == 'ENG':
    region_name = 'Englishman'
elif watershed == 'MV':
    region_name = 'Metro Vancouver'
# RAIN-SNOW THRESHOLD (according to Jennings et al., 2018)
if watershed == 'TSI':
    rain_snow_threshold=0.98 
elif watershed == 'CRU':
    rain_snow_threshold=0.90
elif watershed == 'ENG':
    rain_snow_threshold=0.91
elif watershed == 'MV':
    rain_snow_threshold=0.97

#region Field data
## Import and format field snowplot and snowcourse data------------------------------------------------------------------------------------------------
os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/')
plots=pd.read_csv('snowplots.csv')
plots=plots[plots['region_name']==region_name]
plots['datetime'] = plots['sample_timestamp'].astype('datetime64[ns]')
plots['date']=plots['datetime'].dt.date
easting, northing, zone_number, zone_letter = utm.from_latlon(
    plots['latitude'].values, 
    plots['longitude'].values)
plots['easting']=easting
plots['northing']=northing
plots['year']=plots['datetime'].dt.year
plots['water_year']=np.where(plots['datetime'].dt.month.isin(np.arange(1,9)),
                             plots['year'], plots['year']+1)
plots['coord_water_year']=plots['coord_year']
plots['site_name']=plots['plot_name']
a=plots[plots['sample_type']=='Density']
b=plots[plots['sample_type']=='Depth']
c=pd.merge(a.drop(columns=['latitude','longitude','easting','northing','probe_depth_cm']), # Merge associated (same location, same datetime) Density and Depth measurements. 
           b[['datetime','direction','distance','probe_depth_cm','easting','northing']], 
           on=['datetime','direction','distance'], how='left')
d=c[c['probe_depth_cm'].isna()]
e=pd.merge(d[['id']],a[['id','easting','northing']], on='id', how='left') # We use the coordinates of the Depth measurements. If no Depth meas is available, we use the core depth of the density meas.
plots=pd.concat([pd.merge(e,d.drop(columns=['easting','northing']), on='id', how='inner'),
             c[c['probe_depth_cm'].notna()]],
             ignore_index=True)
plots['snow_depth_m']=plots['probe_depth_cm'].combine_first(plots['core_depth_cm'])/100
plots['sample_type'] = 'snowplot'
plots['core_number'] = np.nan
plots=plots[['id','sub_id','date','datetime','water_year','coord_water_year','region_name','sample_type',
             'site_name','easting','northing','direction', 'distance','core_number',
             'snow_depth_m','density_percent','swe_cm','elevation_m','probe_depth_cm','notes' ]]
del a,b,c,d,e

courses=pd.read_csv('snowcourse.csv')
courses=courses[courses['region_name']==region_name]
courses['datetime'] = courses['date'].astype('datetime64[ns]')
courses['date']=courses['datetime'].dt.date
courses=courses[courses['latitude'].notna()]
courses=courses[courses['density_percent'].notna()]
easting, northing, zone_number, zone_letter = utm.from_latlon(
    courses['latitude'].values, 
    courses['longitude'].values)
courses['easting']=easting
courses['northing']=northing
courses['sample_type'] = 'snowcourse'
courses['site_name']=courses['course_name']
courses['direction']=np.nan
courses['distance']=np.nan
courses['probe_depth_cm']=np.nan
courses['notes']=np.nan
courses['snow_depth_m']=courses['core_depth_cm']/100
courses=courses[['id','sub_id','date','datetime','water_year','coord_water_year','region_name','sample_type',
             'site_name','easting','northing','direction','distance','core_number',
             'snow_depth_m','density_percent','swe_cm','elevation_m','probe_depth_cm','notes' ]]

## QAQC field snowplot and snowcourse data
plots['flag'] = 'AV'
plots.loc[(plots['probe_depth_cm'].isna()), 'flag'] = 'snow depth from core'
plots.loc[(plots['water_year']!=plots['coord_water_year']), 'flag'] = 'coord uncertainty'
plots.loc[(plots['density_percent']>0.8)|(plots['density_percent']<0.1), 'flag'] = 'out of range'
#filt.loc[(filt['snow_depth']-filt['core_depth']<-5)|(filt['snow_depth']/filt['core_depth']>=2), 'flag'] = 'core'
plot_filt=plots[(plots['flag']=='AV')|(plots['flag']=='snow depth from core')]
courses['flag'] = 'AV'
courses.loc[(courses['water_year']!=courses['coord_water_year']), 'flag'] = 'coord uncertainty'
courses.loc[(courses['density_percent']>80)|(courses['density_percent']<10), 'flag'] = 'out of range'
cours_filt=courses[(courses['flag']=='AV')]
filt=pd.concat([plot_filt, cours_filt], ignore_index=True)

# Get centre coordinates and average densities and snowdepths by plot or by course
plot_gr=filt[filt['sample_type']=='snowplot']
plot_gr=plot_gr[['easting','northing','direction','distance','site_name','date','water_year','snow_depth_m','density_percent','swe_cm']]
plot_gr['easting']=np.where((plot_gr['direction']=='Centre')|(plot_gr['direction']=='S')|(plot_gr['direction']=='N'),plot_gr['easting'],
                                np.where(plot_gr['direction']=='W',(plot_gr['easting']-plot_gr['distance']),
                                     np.where(plot_gr['direction']=='E',(plot_gr['easting']+plot_gr['distance']),
                                              np.where((plot_gr['direction']=='SE')|(plot_gr['direction']=='NE'),(plot_gr['easting']+(plot_gr['distance']*0.7)),
                                                       np.where((plot_gr['direction']=='SE')|(plot_gr['direction']=='NE'),(plot_gr['easting']-(plot_gr['distance']*0.7)),np.nan)))))
plot_gr['northing']=np.where((plot_gr['direction']=='Centre')|(plot_gr['direction']=='W')|(plot_gr['direction']=='E'),plot_gr['northing'],
                                np.where(plot_gr['direction']=='S',(plot_gr['northing']-plot_gr['distance']),
                                     np.where(plot_gr['direction']=='N',(plot_gr['northing']+plot_gr['distance']),
                                              np.where((plot_gr['direction']=='NE')|(plot_gr['direction']=='NW'),(plot_gr['northing']+(plot_gr['distance']*0.7)),
                                                       np.where((plot_gr['direction']=='SE')|(plot_gr['direction']=='SW'),(plot_gr['northing']-(plot_gr['distance']*0.7)),np.nan)))))
plot_gr=plot_gr.groupby(['site_name','date']).mean().reset_index()
plot_gr=plot_gr.drop(columns='distance')
plot_gr['sample_type'] = 'snowplot'

course_gr=filt[filt['sample_type']=='snowcourse']
course_gr=course_gr[['easting','northing','site_name','date','water_year','snow_depth_m','density_percent','swe_cm']]
course_gr=course_gr.groupby(['site_name','date']).mean().reset_index()
course_gr['sample_type'] = 'snowcourse'
grouped=pd.concat((plot_gr,course_gr),ignore_index=True)
grouped['water_year']=grouped['water_year'].astype(int)

# Save plots data
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
filt.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'.csv',index=False)
grouped.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'_grouped.csv',index=False)
#endregion

#region Xt processing
## Get Xt for every 1m elevation band, every 'day in season' (day since Sep 1st) of each water year
grouped['day_in_season']=grouped['date']-pd.to_datetime((grouped['water_year']-1).astype('string')+"-09-01 12:00:00").dt.date
grouped['day_in_season']=grouped['day_in_season'].dt.days

# Import bare earth and get min and max elev
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(extent)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif', bands='all'))
nans=np.where(BE<0)
BE[nans]=np.nan
BE=BE.astype('float64')
del nans
min_elev=np.nanmin(BE)
min_elev=np.floor(min_elev).astype(int)
max_elev=np.nanmax(BE)
max_elev=np.ceil(max_elev).astype(int)

# Import station elevations
os.chdir(str(drive)+':/LiDAR_data_processing/Weather_station_data/'+str(watershed)+'/')
WSup_airT_elev=int(np.array(pd.read_csv('Metadata_'+str(watershed)+'.csv', usecols=['WSup_airT_elev']))[0])
WSlow_airT_elev=int(np.array(pd.read_csv('Metadata_'+str(watershed)+'.csv', usecols=['WSlow_airT_elev']))[0])
WSup_precip_elev=int(np.array(pd.read_csv('Metadata_'+str(watershed)+'.csv', usecols=['WSup_precip_elev']))[0])
WSlow_precip_elev=int(np.array(pd.read_csv('Metadata_'+str(watershed)+'.csv', usecols=['WSlow_precip_elev']))[0])

# Import weather station data
Tair_up = np.array(pd.read_csv('WS_data_'+str(watershed)+'.csv', usecols=['Tair_up']))
Tair_low = np.array(pd.read_csv('WS_data_'+str(watershed)+'.csv', usecols=['Tair_low']))
precip_pipe = np.array(pd.read_csv('WS_data_'+str(watershed)+'.csv', usecols=['PC_up_mm']))
DateTime=np.array(pd.read_csv('WS_data_'+str(watershed)+'.csv',usecols=['DateTime']))

# Reformat total precip data
precip_pipe=precip_pipe.astype('float64')
df=pd.DataFrame(precip_pipe) 
Precip=np.array(df).flatten()
del x,n,df,precip_pipe

# Calculations -----------------------------------------------------------------------------
# Calculate atmospheric lapse rate for every time interval
lapse_rate = (Tair_up-Tair_low)/(WSup_airT_elev-WSlow_airT_elev)
del WSlow_airT_elev,Tair_low

# Calculate air temperature for each elevation and each timestamp based on atmospheric lapse rate
# Set all negative airTemps to 0 so that PDH (positive degree hour) sum can be calculated
elevs = np.arange(start=min_elev, stop=max_elev, step=1) 
Tair_corrected=[]
PDH_all=[]
for n in range(len(elevs)):
    x=(elevs[n]-WSup_airT_elev)*lapse_rate
    Corrected=Tair_up+x
    PDH=Corrected
    non_PDH=np.where(Corrected<0)
    PDH[non_PDH] = 0
    Tair_corrected.append(Corrected)
    PDH_all.append(PDH)

# Calculate snowfall for each elevation and each timestamp
Snowfall=[]
for n in range(len(elevs)):
    snow_vs_rain=Tair_corrected[n]
    snow_vs_rain=np.reshape(snow_vs_rain,(len(snow_vs_rain),))
    non_snow=np.where(snow_vs_rain>rain_snow_threshold)
    snow=np.where(snow_vs_rain<=rain_snow_threshold)
    snow_vs_rain[non_snow]=0
    snow_vs_rain[snow]=1
    snowfall=np.multiply(snow_vs_rain,Precip)
    Snowfall.append(snowfall)

# Sort date and time data and group with results
Dates=DateTime.astype('U8')
Dates = np_f.replace(Dates, '/', '')
x=Dates.astype(float)
Dates_all=[]
for n in range(len(temp_diffs)):
    Dates_all.append(x)
Snowfall2=[]
for n in range(len(Snowfall)):
    s=np.reshape(Snowfall[n],(len(Snowfall[n]),1))
    Snowfall2.append(s)
Snowfall=Snowfall2
all_data=[]
for n in range(len(temp_diffs)):
    joined=np.concatenate((Dates_all[n],PDH_all[n],Snowfall[n]),axis=1)
    all_data.append(joined)

# Find all unique dates and their indices
date_test_all=[]
indice_test_all=[]
[date_test,indice_test]=np.unique(x, return_index=True)
date_test_sorted=(x[np.sort(indice_test)])
indices_all=[]
for y in range(len(date_test_sorted)):
    [indices,extracol]=np.where(x==date_test_sorted[y])
    indices_all.append(indices)

# Extract PHD data for every unique date, for every elevation band
sorted_PDH_all=[]
for x in range(len(temp_diffs)):
    PDH_test=np.ndarray.flatten(PDH_all[x])
    sorted_PDH=[]
    for y in range(len(indices_all)):
        ind_test=indices_all[y]
        PDH_test2=PDH_test[ind_test]
        sorted_PDH.append(PDH_test2)
    sorted_PDH_all.append(sorted_PDH)
    
# Extract snowfall data for every unique date, for every elevation band
sorted_snowfall_all=[]
for x in range(len(temp_diffs)):
    snowfall_test=np.ndarray.flatten(Snowfall[x])
    sorted_snowfall=[]
    for y in range(len(indices_all)):
        ind_test=indices_all[y]
        snowfall_test2=snowfall_test[ind_test]
        sorted_snowfall.append(snowfall_test2)
    sorted_snowfall_all.append(sorted_snowfall)

# Find mean daily PDD for each elevation band
PDD_daily_allsites=[]
for z in range(len(temp_diffs)):
    #PDD_daily_test=[sum(x)/len(x) for x in sorted_PDH_all[z]] # having divide by 0 issue here
    PDD_daily_test=[np.mean(vals) if len(vals) > 0 else np.nan for vals in sorted_PDH_all[z]]
    PDD_daily_allsites.append(PDD_daily_test)

# Find total daily snowfall for elevation band
snowfall_daily_allsites=[]
for z in range(len(temp_diffs)):
    snowfall_daily_test=[sum(x) for x in sorted_snowfall_all[z]]
    snowfall_daily_allsites.append(snowfall_daily_test)

# Extract PDD data prior to each of the phases (since preceding September 1st)
Pre_survey_PDD_all=[]
for m in range(len(temp_diffs)):
    Pre_survey_PDD=[]
    PDD_x=PDD_daily_allsites[m]
    for n in range(len(survey_dates)):
        PDD_pre=PDD_x[0:survey_dates[n]]
        PDD_pre=np.array(PDD_pre)
        nans=np.argwhere(np.isnan(PDD_pre))
        PDD_pre[nans]=0
        Pre_survey_PDD.append(PDD_pre)
    Pre_survey_PDD_all.append(Pre_survey_PDD)

# Extract snowfall data prior to each of the phases (since preceding September 1st)
Pre_survey_S_all=[]
for m in range(len(temp_diffs)):
    Pre_survey_S=[]
    S_x=snowfall_daily_allsites[m]
    for n in range(len(survey_dates)):
        S_pre=S_x[0:survey_dates[n]]
        S_pre=np.array(S_pre)
        nans=np.argwhere(np.isnan(S_pre))
        S_pre[nans]=0
        Pre_survey_S.append(S_pre)
    Pre_survey_S_all.append(Pre_survey_S)

# Find reverse cumulative PDD sum
reverse_cumulative_PDD_allsites=[]
for m in range(len(temp_diffs)):
    reverse_cumulative_PDD=[]
    PS_PDD=Pre_survey_PDD_all[m]
    for n in range(len(survey_dates)):
        PS_PDD_x=PS_PDD[n]
        reverse_cumulative_PDD_sum=np.cumsum(PS_PDD_x[::-1])[::-1] 
        reverse_cumulative_PDD.append(reverse_cumulative_PDD_sum)
    reverse_cumulative_PDD_allsites.append(reverse_cumulative_PDD)

# Extract PDD data prior to each of the phases (since preceding September 1st)
Pre_survey_PDD_all=[]
for m in range(len(temp_diffs)):
    Pre_survey_PDD=[]
    PDD_x=PDD_daily_allsites[m]
    for n in range(len(survey_dates)):
        PDD_pre=PDD_x[0:survey_dates[n]]
        PDD_pre=np.array(PDD_pre)
        nans=np.argwhere(np.isnan(PDD_pre))
        PDD_pre[nans]=0
        Pre_survey_PDD.append(PDD_pre)
    Pre_survey_PDD_all.append(Pre_survey_PDD)

# Find total snowfall and PDD sum for each phase and elevation
total_PDD_sum=[]
total_S_sum=[]
for m in range(len(temp_diffs)):
    total_PDD=[]
    total_S=[]
    PS_PDD=Pre_survey_PDD_all[m]
    PS_S=Pre_survey_S_all[m]
    for n in range(len(survey_dates)):
        PS_PDD_x=PS_PDD[n]
        PS_S_x=PS_S[n]
        PDD_sum=np.nansum(PS_PDD_x)
        S_sum=np.nansum(PS_S_x)
        total_PDD.append(PDD_sum)
        total_S.append(S_sum)
    total_PDD_sum.append(total_PDD)
    total_S_sum.append(total_S)

# Find snow-settling metric (per day)
MMM=[]
for m in range(len(temp_diffs)):
    PDD_xx=reverse_cumulative_PDD_allsites[m]
    S_xx=Pre_survey_S_all[m]
    MM=[]
    for n in range(len(survey_dates)):
        PDD_yy=PDD_xx[n]
        S_yy=S_xx[n]
        np.seterr(divide='ignore',invalid='ignore')
        M=np.divide(S_yy,PDD_yy)
        MM.append(M)
    MMM.append(MM)

# Find snow settling-metric (total for each phase, for each elevation)
metric=[]
for z in range(len(temp_diffs)):
    MMMM=[sum(x)/len(x) for x in MMM[z]]
    np.seterr(divide='ignore',invalid='ignore')
    ones=np.divide(MMMM,MMMM)
    MMMM=np.divide(ones,MMMM)
    metric.append(MMMM)

# Output ---------------------------------------------------------------------------------------------------
Xt=np.array(metric)
PDD_sum=np.array(total_PDD_sum)
Total_snowfall=np.array(total_S_sum)
Xt2=[]
for n in range(len(Xt)):
    x=Xt[n]
    nans=np.argwhere(np.isnan(x))
    x[nans]=0
    Xt2.append(x)
Xt=np.array(Xt2)
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output')
Xt_save=pd.DataFrame(Xt,columns=phases)
Xt_save.insert(0, "Elevation",elevs)
Xt_save.to_csv("Xt_1m_intervals_"+str(extent)+"_"+str(year)+".csv", index = False)
PDD_sum_save=pd.DataFrame(PDD_sum,columns=phases)
PDD_sum_save.insert(0, "Elevation",elevs)
PDD_sum_save.to_csv("PDD_1m_intervals_"+str(extent)+"_"+str(year)+".csv", index = False)
Total_snowfall_save=pd.DataFrame(Total_snowfall,columns=phases)
Total_snowfall_save.insert(0, "Elevation",elevs)
Total_snowfall_save.to_csv("S_1m_intervals_"+str(extent)+"_"+str(year)+".csv", index = False)

Xt_plot = Xt_save.plot(kind='line',x='Elevation',color=['red','blue','green'])
Xt_plot.set_ylabel('Xt')
Xt_plot.figure.savefig('Xt_1m_intervals_'+str(extent)+'_'+str(year)+'.png')
PDD_plot = PDD_sum_save.plot(kind='line',x='Elevation',color=['red','blue','green'])
PDD_plot.set_ylabel('PDD')
PDD_plot.figure.savefig('PDD_1m_intervals_'+str(extent)+'_'+str(year)+'.png')
Total_snowfall_plot = Total_snowfall_save.plot(kind='line',x='Elevation',color=['red','blue','green'])
Total_snowfall_plot.set_ylabel('Total snowfall')
Total_snowfall_plot.figure.savefig('Total_snowfall_1m_intervals_'+str(extent)+'_'+str(year)+'.png')


#endregion





## Import LiDAR derived input variables and sample for GROUPED locations
file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/Input_variables_'+'v'+str(DENSversion)+'_cardinal.csv')
if file.is_file():
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
    grouped=pd.read_csv('Input_variables_'+'v'+str(DENSversion)+'_cardinal.csv')
    grouped['year'] = grouped['year'].astype(str)
else:
    buffer_distance = 10 # buffer distance around centre point of cardinal plot
    # Import Bare Earth metrics 
    s_BE=[]
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/')
    be_list = ['DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CD_v'+str(CANversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CC_v'+str(CANversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CH_v'+str(CANversion)+'_'+str(resolution1)+'m.tif']
    for a in be_list:
        b=[]
        src = rasterio.open(a)
        for x in range(len(grouped)):
            coords = (grouped.at[x,'easting_m'], grouped.at[x,'northing_m'])
            point = Point(coords)
            buffer_poly = point.buffer(buffer_distance)
            # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
            mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src.shape,transform=src.transform,invert=False)
            data = src.read(1) # Read data and apply mask. Read first band
            sampled_values = data[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
            sampled_value = np.nanmean(sampled_values) # Take the mean of all pixels within buffer
            b.append(sampled_value)
        s_BE.append(b)

    BE_inputs=['elevation_lidar','slope_lidar','curvature_lidar','northness_lidar','eastness_lidar','canopy_density_lidar','canopy_cover_lidar','canopy_height_lidar']
    for x in range(len(BE_inputs)):
        grouped[BE_inputs[x]] = s_BE[x]

    

# Prepare testing and training data for RF model
y=np.array(grouped_final['density'])
variables=[]
for n in range(len(predictors)):
    x = np.array(grouped_final[predictors[n]])
    variables.append(x)

# Normalise input data
all_scalers=[]
for n in range(len(variables)):
    x=variables[n]
    x=x.reshape(len(x),1)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(x)
    X_normalized=X_normalized.reshape(len(X_normalized),)
    variables[n]=X_normalized
    all_scalers.append(scaler)
    
# Reformat data
X=np.transpose(variables)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)

# Create RF model
param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 100)],
              'max_depth': [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)],
              'min_samples_split':[int(x) for x in np.linspace(start = 2, stop = 30, num = 29)],
              'min_samples_leaf':[int(x) for x in np.linspace(start = 1, stop = 10, num = 10)],
              'max_features':[0.3],
              'bootstrap':[True,False]}

model_rf = RandomForestRegressor()
rf_RandomGrid = RandomizedSearchCV(estimator = model_rf, param_distributions = param_grid, verbose=2, n_jobs = -1, n_iter=500, random_state=12345)
rf_RandomGrid.fit(X_train, Y_train)
score_train=rf_RandomGrid.score(X_train,Y_train)
score_test=rf_RandomGrid.score(X_test,Y_test)
hyperparams=rf_RandomGrid.best_params_
pred_test_rf = rf_RandomGrid.predict(X_test)
rmse_rf=np.sqrt(mean_squared_error(Y_test,pred_test_rf))
syst_error=np.mean(pred_test_rf-Y_test)
rand_error=np.std(pred_test_rf-Y_test)
model_rf = rf_RandomGrid.best_estimator_

# Plot an example decision tree
tree_to_plot = model_rf.estimators_[1]
plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=predictors, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()

# Extract feature importances
importances = model_rf.feature_importances_
feature_names = predictors
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Output ---------------------------------------------------------------------------------------------------------------
# Save plots data, filtered plots data, and model input variables (plots data linked to lidar metrics)
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
grouped_final.to_csv('Input_variables_'+'v'+str(DENSversion)+'.csv',index=False)

# Save RF model hyperparameters, error values, and processing variables
joblib.dump(model_rf, 'RF_density_model_'+str(watershed)+'_v'+str(DENSversion)+'.joblib')
hyperparams=pd.DataFrame([hyperparams])
hyperparams.to_csv('Hyperparameters_'+'v'+str(DENSversion)+'.csv',index=False)
feature_importance_df.to_csv('Feature_importance_'+'v'+str(DENSversion)+'.csv',index=False)
all_errors= pd.DataFrame({'rmse':[rmse_rf],'syst_error':[syst_error],'rand_error':[rand_error],'R2_train':[score_train],'R2_test':[score_test]})
all_errors.to_csv('model_error_values_v'+str(DENSversion)+'.csv')
var.to_csv(str(watershed)+'_ML_model_processing_variables_v'+str(DENSversion)+'.csv')

# Export normalization scalers
for n in range(len(all_scalers)):
    scaler=all_scalers[n]
    joblib.dump(scaler, 'scaler'+str(n+1)+'.pkl')

# Plot regressions between predictor variables and snow density (dependent variable)
# Calculate R-squared using scipy.stats.linregress
lst = ['year','phase','easting_m','northing_m','density']
lst.extend(predictors)
final_long=grouped_final[lst]
final_long=final_long.melt(id_vars=['year','phase','easting_m','northing_m','density'],
                      var_name='variable',value_name='value')
variables=final_long['variable'].unique()
R2s=[]
for n in (variables):
    x = final_long[final_long['variable']==n]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x["value"], x["density"])
    r2 = r_value**2
    r2_text = f'$R^2$ = {r2:.3f}' # Format R-squared to 3 decimal places
    R2s.append(r2_text)
g = sns.lmplot(
    data=final_long,      # Your DataFrame
    x="value",            # X-axis variable
    y="density",
    #hue="season",     # Y-axis variable
    col="variable",       # Variable for columns (facets)
    col_wrap=4,
    sharex=False,                
    height=4,             # Height of each facet in inches
    aspect=1.2,           # Aspect ratio of each facet
    line_kws={'color': 'red'}) # Keyword arguments for the regression line
for n in range(len(variables)):
    ax=g.axes[n]
    ax.text(0.05, 0.9, R2s[n], transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Density_modelling/{watershed}/ML_density_model/v{DENSversion}/"
    f"Regression_plots_v{DENSversion}.png",bbox_inches='tight', pad_inches=0.1)

