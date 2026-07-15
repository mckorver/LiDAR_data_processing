
# Import packages
import os
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import utm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import rasterio.features
from shapely.geometry import Point
from shapely.geometry import LineString
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree

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
predictors = []
append_fun(predictors,'predictors')
if watershed == 'TSI':
    region_name = 'TSITIKA' 
elif watershed == 'CRU':
    region_name = 'CRUICKSHANK'
elif watershed == 'ENG':
    region_name = 'ENGLISHMAN'
elif watershed == 'MV':
    region_name = 'METRO VANCOUVER'
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
plots=pd.read_csv('snowplot.csv')
plots=plots[plots['region_name']==region_name]
plots['datetime'] = plots['datetime'].astype('datetime64[ns]')
plots['date']=plots['datetime'].dt.date
plots['year']=plots['datetime'].dt.year
plots['water_year']=np.where(plots['datetime'].dt.month.isin(np.arange(1,9)),
                             plots['year'], plots['year']+1)
easting, northing, zone_number, zone_letter = utm.from_latlon(
    plots['latitude'].values, 
    plots['longitude'].values)
plots['easting']=easting
plots['northing']=northing
plots=plots[plots['distance']<=10] # Remove samples that were taken > 10m from the centre of the plot
plots['site_name']=plots['plot_name']
a=plots[plots['sample_type']=='Density']
b=plots[plots['sample_type']=='Depth']
c=pd.merge(a.drop(columns=['latitude','longitude','easting','northing','probe_depth_cm']), # Merge associated (same location, same datetime) Density and Depth measurements. 
           b[['datetime','direction','distance','probe_depth_cm','easting','northing']], 
           on=['datetime','direction','distance'], how='left')
d=c[c['probe_depth_cm'].isna()] # Set the density measurements with no associated depth measurement aside
e=pd.merge(d[['id']],a[['id','easting','northing']], on='id', how='left') # We use the coordinates of the Depth measurements. If no Depth meas is available, we use the core depth of the density meas.
plots=pd.concat([pd.merge(e,d.drop(columns=['easting','northing']), on='id', how='inner'),
             c[c['probe_depth_cm'].notna()]],
             ignore_index=True)
plots['snow_depth_m']=plots['probe_depth_cm'].combine_first(plots['core_depth_cm'])/100
plots['sample_type'] = 'snowplot'
plots['core_number'] = np.nan
plots['easting10'] = np.nan
plots['northing10'] = np.nan
plots=plots[['id','sub_id','date','datetime','water_year','coord_water_year','region_name','sample_type',
             'site_name','easting','northing','direction', 'distance','easting10','northing10','core_number',
             'snow_depth_m','density_frac','swe_cm','elevation_m','probe_depth_cm','sample_rating','retrieval_percent','notes' ]]
del a,b,c,d,e

courses=pd.read_csv('snowcourse.csv')
loc=pd.read_csv('snowcourse_endpoints.csv')
courses=courses[courses['region_name']==region_name]
courses['datetime'] = courses['datetime'].astype('datetime64[ns]')
courses['date']=courses['datetime'].dt.date
courses=courses[courses['latitude'].notna()]
courses=courses[courses['density_frac'].notna()]
courses=pd.merge(courses,loc,on='course_name',how='left')
courses['sample_type'] = 'snowcourse'
courses['site_name']=courses['course_name']
courses['direction']=np.nan
courses['distance']=np.nan
courses['probe_depth_cm']=np.nan
courses['notes']=np.nan
courses['snow_depth_m']=courses['core_depth_cm']/100
courses=courses[['id','sub_id','date','datetime','water_year','coord_water_year','region_name','sample_type',
             'site_name','easting','northing','direction','distance','easting10','northing10','core_number',
             'snow_depth_m','density_frac','swe_cm','elevation_m','probe_depth_cm','notes' ]]

## QAQC field snowplot and snowcourse data
plots[['flag1','flag2','flag3','flag4','flag5']] = 'AV'
plots.loc[(plots['probe_depth_cm'].isna()), 'flag1'] = 'snow depth from core'
plots.loc[(plots['water_year']!=plots['coord_water_year']), 'flag2'] = 'coord uncertainty'
plots.loc[(plots['density_frac']>0.65)|(plots['density_frac']<0.15), 'flag3'] = 'out of range'
plots.loc[(plots['retrieval_percent'].notna()) & (plots['retrieval_percent']<70), 'flag4'] = 'low retrieval percent'
plots.loc[(plots['sample_rating'].notna()) & (plots['sample_rating']<4), 'flag5'] = 'low sample rating'
plot_filt=plots[(plots['flag3']=='AV')&(plots['flag4']=='AV')&(plots['flag5']=='AV')]
courses[['flag1','flag2']] = 'AV'
courses.loc[(courses['water_year']!=courses['coord_water_year']), 'flag1'] = 'coord uncertainty'
courses.loc[(courses['density_frac']>0.65)|(courses['density_frac']<0.15), 'flag2'] = 'out of range'
cours_filt=courses[courses['flag2']=='AV']
filt=pd.concat([plot_filt, cours_filt], ignore_index=True)

# Get average densities and snowdepths by plot or by course
plot_gr=filt[filt['sample_type']=='snowplot']
plot_gr=plot_gr[['easting','northing','easting10','northing10','site_name','date','water_year','snow_depth_m','density_frac','swe_cm']]
plot_gr=plot_gr.groupby(['site_name','date']).mean().reset_index()
plot_gr['sample_type'] = 'snowplot'

course_gr=filt[filt['sample_type']=='snowcourse']
course_gr=course_gr[['easting','northing','easting10','northing10','site_name','date','water_year','snow_depth_m','density_frac','swe_cm']]
course_gr=course_gr.groupby(['site_name','date']).mean().reset_index()
course_gr['sample_type'] = 'snowcourse'
grouped=pd.concat((plot_gr,course_gr),ignore_index=True)
grouped['water_year']=grouped['water_year'].astype(int)
grouped=grouped.sort_values(by='date')

# Save plots data
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
plots.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'_unfiltered_plots.csv',index=False)
courses.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'_unfiltered_courses.csv',index=False)
filt.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'.csv',index=False)
grouped.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'_grouped.csv',index=False)
#endregion

#region Sample LiDAR derived input variables for field locations
# If an input variables file already exists, skip this whole section.
file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/Input_variables_'+'v'+str(DENSversion)+'.csv')
if file.is_file():
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
    dt=pd.read_csv('Input_variables_'+'v'+str(DENSversion)+'.csv')
else:
    # Import BE and Canopy raster file locations
    os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/')
    be_list = ['DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CD_v'+str(CANversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CC_v'+str(CANversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CH_v'+str(CANversion)+'_'+str(resolution1)+'m.tif']
    # Sample for PLOT locations
    dt_p=grouped[grouped['sample_type']=='snowplot']
    dt_p=dt_p.reset_index(drop=True)
    buffer_p = 10 # buffer distance around centre point of cardinal plot
    BE_p=[]
    for a in be_list:
        b=[]
        src = rasterio.open(a)
        for x in range(len(dt_p)):
            coords = (dt_p.at[x,'easting'], dt_p.at[x,'northing'])
            point = Point(coords)
            buffer_poly = point.buffer(buffer_p)
            # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
            mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src.shape,transform=src.transform,invert=False)
            data = src.read(1) # Read data and apply mask. Read first band
            sampled_values = data[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
            sampled_value = np.nanmedian(sampled_values) # Take the median of all pixels within buffer
            b.append(sampled_value)
        BE_p.append(b)
    del a,b
    # Sample for COURSE locations
    dt_c=grouped[grouped['sample_type']=='snowcourse']
    dt_c = dt_c.dropna(subset=['easting10'])
    dt_c=dt_c.reset_index(drop=True)
    buffer_c = 2 # buffer distance around centre point of cardinal plot
    BE_c=[]
    for a in be_list:
        b=[]
        src = rasterio.open(a)
        for x in range(len(dt_c)):
            coords = [(dt_c.at[x,'easting'], dt_c.at[x,'northing']),(dt_c.at[x,'easting10'], dt_c.at[x,'northing10'])]
            line = LineString(coords)
            buffer_poly = line.buffer(buffer_c)
            # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
            mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src.shape,transform=src.transform,invert=False)
            data = src.read(1) # Read data and apply mask. Read first band
            sampled_values = data[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
            sampled_value = np.nanmedian(sampled_values) # Take the median of all pixels within buffer
            b.append(sampled_value)
        BE_c.append(b)
    del a,b
    
    # Put PLOTS and COURSES data together
    BE_inputs=['elevation_lidar','slope_lidar','curvature_lidar','northness_lidar','eastness_lidar','canopy_density_lidar','canopy_cover_lidar','canopy_height_lidar']
    for x in range(len(BE_inputs)):
        dt_p[BE_inputs[x]] = BE_p[x]
        dt_c[BE_inputs[x]] = BE_c[x]
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
    dt=pd.concat([dt_p, dt_c], ignore_index=True)
    
    # Associate Xt value by date and elevation
    dt_Xt=pd.read_csv(str(drive)+':/LiDAR_data_processing/Weather_station_data/'+str(watershed)+'/Xt_daily_elevation.csv')
    dt['elevation_lidar']=dt['elevation_lidar'].round(0).astype(int)
    dt['elevation_str']=dt['elevation_lidar'].astype(str)
    dt['date']=dt['date'].astype('datetime64[ns]')
    dt=dt.sort_values(by='date')
    dt_Xt['date']=dt_Xt['date'].astype('datetime64[ns]')
    start_date=dt_Xt['date'].iloc[0]
    dt=dt[dt['date']>=start_date]
    dt=dt.reset_index(drop=True)
    Xt=[]
    for n in range(len(dt)):
        a = dt_Xt.loc[dt_Xt['date']==dt['date'][n],dt['elevation_str'][n]].item()
        Xt.append(a)
    dt['Xt_model']=Xt
    dt=dt.drop(columns='elevation_str')

    # Add 'day in season'
    dt['month']=dt['date'].dt.month
    dt['year']=dt['date'].dt.year
    dt1=dt[dt['month']<9].copy()
    dt1['day1']=pd.to_datetime((dt1['year']-1).astype(str) + "-09-01")
    dt2=dt[dt['month']>=9].copy()
    dt2['day1']=pd.to_datetime((dt2['year']).astype(str) + "-09-01")
    dt=pd.concat([dt1,dt2], ignore_index=True)    
    dt=dt.sort_values(by='date')
    dt['day_in_season'] = dt['date'] - dt['day1']
    dt['day_in_season'] = dt['day_in_season'].dt.total_seconds()/86400
    dt=dt.drop(columns=['month','year','day1'])

    dt.to_csv('Input_variables_'+'v'+str(DENSversion)+'.csv',index=False)
#endregion

#region Train model
# Prepare testing and training data for RF model
y=np.array(dt['density_frac'])
variables=[]
for n in range(len(predictors)):
    x = np.array(dt[predictors[n]])
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
# Save RF model hyperparameters, error values, and processing variables
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
joblib.dump(model_rf, 'RF_density_model_'+str(watershed)+'_v'+str(DENSversion)+'.joblib')
hyperparams=pd.DataFrame([hyperparams])
hyperparams.to_csv('Hyperparameters_'+'v'+str(DENSversion)+'.csv',index=False)
feature_importance_df.to_csv('Feature_importance_'+'v'+str(DENSversion)+'.csv',index=False)
all_errors= pd.DataFrame({'rmse':[rmse_rf],'syst_error':[syst_error],'rand_error':[rand_error],'R2_train':[score_train],'R2_test':[score_test]})
all_errors.to_csv('model_error_values_v'+str(DENSversion)+'.csv')
var.to_csv(str(watershed)+'_ML_model_processing_variables_v'+str(DENSversion)+'.csv',index=False)

# Export normalization scalers
for n in range(len(all_scalers)):
    scaler=all_scalers[n]
    joblib.dump(scaler, 'scaler'+str(n+1)+'.pkl')

# Plot regressions between predictor variables and snow density (dependent variable)
# Calculate R-squared using scipy.stats.linregress
lst = ['date','easting','northing','density_frac']
lst.extend(predictors)
final_long=dt[lst]
final_long=final_long.melt(id_vars=['date','easting','northing','density_frac'],
                      var_name='variable',value_name='value')
variables=final_long['variable'].unique()
R2s=[]
for n in (variables):
    x = final_long[final_long['variable']==n]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x["value"], x["density_frac"])
    r2 = r_value**2
    r2_text = f'$R^2$ = {r2:.3f}' # Format R-squared to 3 decimal places
    R2s.append(r2_text)
g = sns.lmplot(
    data=final_long,      # Your DataFrame
    x="value",            # X-axis variable
    y="density_frac",
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