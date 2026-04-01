
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

# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/Density_modelling/ML_model_processing_variables.csv', dtype={'years':str, 'resolution1':str, 'BEversion':str, 'CANversion':str, 'DENSversion':str})
watershed = var['watershed'][0]
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

## QAQC field data
datetimes_field=[]
datetimes_aco=[]
eastings=[]
northings=[]
plot_ids=[]
plot_types=[]
card_dirs=[]
depths=[]
cores=[]
densities=[]
manual_remove=[]
field_phases=[]
for a in range(len(years)):
    os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/'+str(years[a]))
    for b in range(len(phases[a])):
        # Read field data collection coordinates
        datetime_f=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['plot_datetime'],parse_dates=['plot_datetime']))
        datetime_f=datetime_f.reshape(len(datetime_f),)
        datetime_l=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['aco_datetime'],parse_dates=['aco_datetime']))
        datetime_l=datetime_l.reshape(len(datetime_l),)
        easting=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['easting_m'])).astype('float64')
        easting=easting.reshape(len(easting),)
        northing=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['northing_m'])).astype('float64')
        northing=northing.reshape(len(northing),)
        plot_id=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['plot_id']))
        plot_id=plot_id.reshape(len(plot_id),)
        plot_type=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['plot_type']))
        plot_type=plot_type.reshape(len(plot_type),)
        cardinal=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['cardinal']))
        cardinal=cardinal.reshape(len(cardinal),)
        depth=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['snow_depth'])).astype('float64')
        depth=depth.reshape(len(depth),)
        core=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['core_length_final'])).astype('float64')
        core=core.reshape(len(core),)
        density=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['density'])).astype('float64')
        density=density.reshape(len(density),)
        manual_rem=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['manual_remove']))
        manual_rem=manual_rem.reshape(len(manual_rem),)
        field_phase=np.array(phases[a][b])
        field_phase=np.repeat(field_phase,len(easting))
        datetimes_field.append(datetime_f)
        datetimes_aco.append(datetime_l)
        eastings.append(easting)
        northings.append(northing)
        plot_ids.append(plot_id)
        plot_types.append(plot_type)
        card_dirs.append(cardinal)
        depths.append(depth)
        cores.append(core)
        densities.append(density)
        manual_remove.append(manual_rem)
        field_phases.append(field_phase)

df=pd.DataFrame({"phase":np.concatenate(field_phases),"datetime_field":np.concatenate(datetimes_field),"datetime_aco":np.concatenate(datetimes_aco),
                 "easting_m":np.concatenate(eastings),"northing_m":np.concatenate(northings),
                 "plot_id":np.concatenate(plot_ids),"plot_type":np.concatenate(plot_types),"cardinal":np.concatenate(card_dirs),
                 "snow_depth":np.concatenate(depths),"core_depth":np.concatenate(cores),
                 "density":np.concatenate(densities), "manual_remove":np.concatenate(manual_remove)})
df['time_gap_hr']=df['datetime_field']-df['datetime_aco']
df['time_gap_hr'] = df['time_gap_hr'].dt.total_seconds()/3600
df['time_gap_hr'] = df['time_gap_hr'].abs().astype('int64')
df['year'] = df['datetime_field'].dt.year.astype('string')
df['day1'] = pd.to_datetime(df['year']+"-03-01 12:00:00")
df['day_in_season'] = df['datetime_aco'] - df['day1']
df['day_in_season'] = df['day_in_season'].dt.total_seconds()/86400
df['snow_depth_m'] = df['snow_depth']/100

# QAQC field data
df=df[(df['density'].notna())&(df['density']>0)]
df['flag'] = 'AV'
df.loc[(df['time_gap_hr']>60), 'flag'] = 'time'
df.loc[(df['manual_remove']=='Y'), 'flag'] = 'manual'
df.loc[(df['density']>0.8)|(df['density']<0.1), 'flag'] = 'range'
df.loc[(df['snow_depth']-df['core_depth']<-5)|(df['snow_depth']/df['core_depth']>=2), 'flag'] = 'core'
filt=df.loc[(df['flag']=='AV')]

# Fix some issues with plot types
grouped=filt.drop(columns=['snow_depth','flag','manual_remove','time_gap_hr','core_depth'])
grouped.loc[grouped['plot_type'].isna(), 'plot_type'] = 'Cardinal 10 m'
grouped.loc[grouped['plot_type']=='Cardinal 30 m', 'plot_type'] = 'Cardinal 10 m'
grouped=grouped.loc[(grouped['plot_type']=='Cardinal 10 m')]

# Get centre coordinates
grouped=grouped[['easting_m','northing_m','phase','cardinal','plot_id','year','density','snow_depth_m','day_in_season']]
grouped['easting_m']=np.where(grouped['cardinal']=='W',grouped['easting_m']+10,grouped['easting_m'])
grouped['easting_m']=np.where(grouped['cardinal']=='E',grouped['easting_m']-10,grouped['easting_m'])
grouped['northing_m']=np.where(grouped['cardinal']=='N',grouped['northing_m']-10,grouped['northing_m'])
grouped['northing_m']=np.where(grouped['cardinal']=='S',grouped['northing_m']+10,grouped['northing_m'])
grouped['northing_m']=np.where(grouped['cardinal']=='SW',grouped['northing_m']+10,grouped['northing_m'])
grouped['easting_m']=np.where(grouped['cardinal']=='SW',grouped['easting_m']+10,grouped['easting_m'])
grouped_final=grouped.groupby(['phase','plot_id','year']).mean().reset_index()

# Save field data
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
df.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'.csv',index=False)
filt.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'_filtered.csv',index=False)
grouped.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'_grouped.csv',index=False)
del datetimes_field,datetimes_aco,eastings,northings,depths,cores,densities,manual_remove,field_phases

## Import LiDAR derived input variables and sample for field locations
file=Path(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/Input_variables_'+'v'+str(DENSversion)+'.csv')
if file.is_file():
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
    grouped_final=pd.read_csv('Input_variables_'+'v'+str(DENSversion)+'.csv')
    grouped_final['year'] = grouped_final['year'].astype(str)
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
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CD_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CC_v'+str(BEversion)+'_'+str(resolution1)+'m.tif',
            'Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/'+str(watershed)+'_CH_v'+str(BEversion)+'_'+str(resolution1)+'m.tif']
    for a in be_list:
        b=[]
        src = rasterio.open(a)
        for x in range(len(grouped_final)):
            coords = (grouped_final.at[x,'easting_m'], grouped_final.at[x,'northing_m'])
            point = Point(coords)
            buffer_poly = point.buffer(buffer_distance)
            # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
            mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src.shape,transform=src.transform,invert=False)
            data = src.read(1) # Read data and apply mask. Read first band
            sampled_values = data[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
            sampled_value = sampled_values.mean() # Take the mean of all pixels within buffer
            b.append(sampled_value)
        s_BE.append(b)

    BE_inputs=['elevation_lidar','slope_lidar','curvature_lidar','northness_lidar','eastness_lidar','canopy_density_lidar','canopy_cover_lidar','canopy_height_lidar']
    for x in range(len(BE_inputs)):
        grouped_final[BE_inputs[x]] = s_BE[x]

    # Import meteorological parameters
    s_meteo=[]
    meteo_list = ['Distributed_Xt_','Distributed_PDD_','Distributed_Snowfall_']
    for m in range(len(years)):
        list3=[]
        os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(years[m])+'/Output/resolution_'+str(resolution1)+'m/')
        for n in range(len(phases[m])):
            list2=[]
            loc = grouped_final[(grouped_final['year'] == str(years[m])) & (grouped_final['phase'] == str(phases[m][n]))]
            index=loc.index.tolist()
            for b in index:
                list1=[]
                list1.append(b)
                coords = (loc.at[b,'easting_m'], loc.at[b,'northing_m'])
                point = Point(coords)
                buffer_poly = point.buffer(buffer_distance)
                # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
                for a in meteo_list:
                    src = rasterio.open(a+str(watershed)+'_'+str(years[m])+'_'+str(phases[m][n])+'.tif') 
                    mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src.shape,transform=src.transform,invert=False)
                    data = src.read(1) # Read data and apply mask. Read first band
                    sampled_values = data[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
                    sampled_value = sampled_values.mean() # Take the mean of all pixels within buffer
                    list1.append(sampled_value)
                list2.append(list1)
            list3.extend(list2)
        s_meteo.extend(list3)

    pd_meteo=pd.DataFrame(s_meteo,columns=['index','Xt_model','PDD_model','Snowfall_model']).set_index('index')
    grouped_final = pd.merge(grouped_final, pd_meteo, left_index=True, right_index=True, how='inner')
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion)+'/')
    grouped_final.to_csv('Input_variables_'+'v'+str(DENSversion)+'.csv',index=False)

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
rf_RandomGrid = RandomizedSearchCV(estimator = model_rf, param_distributions = param_grid, verbose=2, n_jobs = -1, n_iter=100, random_state=12345)
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
# Save field data, filtered field data, and model input variables (field data linked to lidar metrics)
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

