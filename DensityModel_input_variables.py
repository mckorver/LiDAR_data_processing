
# Import packages
import rasterio
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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
        depths.append(depth)
        cores.append(core)
        densities.append(density)
        manual_remove.append(manual_rem)
        field_phases.append(field_phase)

df=pd.DataFrame({"phase":np.concatenate(field_phases),"datetime_field":np.concatenate(datetimes_field),"datetime_aco":np.concatenate(datetimes_aco),
                 "easting_m":np.concatenate(eastings),"northing_m":np.concatenate(northings),
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
del datetimes_field,datetimes_aco,eastings,northings,depths,cores,densities,manual_remove,field_phases

## Import LiDAR derived input variables and sample for field locations
# Import Bare Earth metrics    
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution1)+'m/')     
elevations = rasterio.open(str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')
slope = rasterio.open(str(watershed)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')
curvature = rasterio.open(str(watershed)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')
aspect = rasterio.open(str(watershed)+'_Aspect_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')
northness = rasterio.open(str(watershed)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')
eastness = rasterio.open(str(watershed)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')

# Import canopy metrics
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution1)+'m/')    
canopy_density = rasterio.open(str(watershed)+'_CD_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')
canopy_cover = rasterio.open(str(watershed)+'_CC_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')
canopy_height = rasterio.open(str(watershed)+'_CH_v'+str(BEversion)+'_'+str(resolution1)+'m.tif')

# Import modelled meteo parameters and sample metrics at field locations
s_elevations=[]
s_slope=[]
s_curvature=[]
s_aspect=[]
s_northness=[]
s_eastness=[]
s_canopyd=[]
s_canopyc=[]
s_canopyh=[]
s_Xt=[]
s_PDD=[]
s_cumsnow=[]
eastings=[]
northings=[]
depths=[]
densities=[]
all_years=[]
all_phases=[]
all_months=[]

for a in range(len(years)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(years[a])+'/Output/resolution_'+str(resolution1)+'m/')
    for b in range(len(phases[a])):
        year=years[a]
        phase=phases[a][b]

        # Read gridded meteo parameters
        xt = rasterio.open('Distributed_Xt_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.tif')
        pdd = rasterio.open('Distributed_PDD_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.tif')
        cumsnow = rasterio.open('Distributed_Snowfall_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.tif')

        # Select field data
        x=filt[(filt['year']==years[a]) & (filt['phase']==phases[a][b])]
        easting=np.array(x['easting_m'])
        northing=np.array(x['northing_m'])
        northing=np.array(x['northing_m'])
        snow=np.array(x['snow_depth_m'])
        density=np.array(x['density'])
        month=np.array(x['month'])

        # Sample gridded datasets at field data collection points
        c=[]
        for m in range(len(easting)):
            cl=(easting[m],northing[m])
            c.append(cl)
            all_years.append(year)
            all_phases.append(phase)
        s_e = [x for x in elevations.sample(c)]
        s_s = [x for x in slope.sample(c)]
        s_c = [x for x in curvature.sample(c)]
        s_a = [x for x in aspect.sample(c)]
        s_no = [x for x in northness.sample(c)]
        s_ea = [x for x in eastness.sample(c)]
        s_cd = [x for x in canopy_density.sample(c)]
        s_cc = [x for x in canopy_cover.sample(c)]
        s_ch = [x for x in canopy_height.sample(c)]
        s_X = [x for x in xt.sample(c)]
        s_P = [x for x in pdd.sample(c)]
        s_Sn = [x for x in cumsnow.sample(c)]
        s_elevations.append(s_e)
        s_slope.append(s_s)
        s_curvature.append(s_c)
        s_aspect.append(s_a)
        s_northness.append(s_no)
        s_eastness.append(s_ea)
        s_canopyd.append(s_cd)
        s_canopyc.append(s_cc)
        s_canopyh.append(s_ch)
        s_Xt.append(s_X)
        s_PDD.append(s_P)
        s_cumsnow.append(s_Sn)
        eastings.append(easting)
        northings.append(northing)
        depths.append(snow)
        densities.append(density)
        all_months.append(month)

sampled_elevations=[]
sampled_slope=[]
sampled_curvature=[]
sampled_aspect=[]
sampled_northness=[]
sampled_eastness=[]
sampled_canopyd=[]
sampled_canopyc=[]
sampled_canopyh=[]
sampled_Xt=[]
sampled_PDD=[]
sampled_cumsnow=[]
for a in range(len(s_elevations)):
    for b in range(len(s_elevations[a])):
        ss_elevations=np.array(s_elevations[a][b]).astype('float64')
        ss_slope=np.array(s_slope[a][b]).astype('float64')
        ss_curvature=np.array(s_curvature[a][b]).astype('float64')
        ss_aspect=np.array(s_aspect[a][b]).astype('float64')
        ss_northness=np.array(s_northness[a][b]).astype('float64')
        ss_eastness=np.array(s_eastness[a][b]).astype('float64')
        ss_canopyd=np.array(s_canopyd[a][b]).astype('float64')
        ss_canopyc=np.array(s_canopyc[a][b]).astype('float64')
        ss_canopyh=np.array(s_canopyh[a][b]).astype('float64')
        ss_Xt=np.array(s_Xt[a][b]).astype('float64')
        ss_PDD=np.array(s_PDD[a][b]).astype('float64')
        ss_Snow=np.array(s_cumsnow[a][b]).astype('float64')
        sampled_elevations.append(ss_elevations)
        sampled_slope.append(ss_slope)
        sampled_curvature.append(ss_curvature)
        sampled_aspect.append(ss_aspect)
        sampled_northness.append(ss_northness)
        sampled_eastness.append(ss_eastness)
        sampled_canopyd.append(ss_canopyd)
        sampled_canopyc.append(ss_canopyc)
        sampled_canopyh.append(ss_canopyh)
        sampled_Xt.append(ss_Xt)
        sampled_PDD.append(ss_PDD)
        sampled_cumsnow.append(ss_Snow)
        
final=pd.DataFrame({"year":all_years,"phase":all_phases,"month":np.concatenate(all_months),
                    "easting_m":np.concatenate(eastings),"northing_m":np.concatenate(northings),
                    "snow_depth_m":np.concatenate(depths),"density_gcm3":np.concatenate(densities),
                    "elevation_lidar":np.concatenate(sampled_elevations),"slope_lidar":np.concatenate(sampled_slope),
                    "curvature_lidar":np.concatenate(sampled_curvature),"aspect_lidar":np.concatenate(sampled_aspect),
                    "eastness_lidar":np.concatenate(sampled_eastness),"northness_lidar":np.concatenate(sampled_northness),
                    "Xt":np.concatenate(sampled_Xt),"PDD_sum":np.concatenate(sampled_PDD),"Total_snowfall":np.concatenate(sampled_cumsnow),
                    "canopy_density":np.concatenate(sampled_canopyd),"canopy_cover":np.concatenate(sampled_canopyc),"canopy_height":np.concatenate(sampled_canopyh)})
final['season']=np.where(np.isin(final['month'], [3,4]),"early","late")
final=final.drop(columns=['aspect_lidar'])

# Prepare testing and training data for RF model
y=np.array(final['density_gcm3'])
variables=[]
for n in range(len(predictors)):
    x = np.array(final[predictors[n]])
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
model_rf = RandomForestRegressor(n_estimators=500,max_depth=5,min_samples_split=20,min_samples_leaf=10,max_features=0.3,n_jobs=-1,random_state=12345)
model_rf.fit(X_train, Y_train)
pred_test_rf = model_rf.predict(X_test)
rmse_rf=np.sqrt(mean_squared_error(Y_test,pred_test_rf))
syst_error=np.mean(pred_test_rf-Y_test)
rand_error=np.std(pred_test_rf-Y_test)

# Output ---------------------------------------------------------------------------------------------------------------
# Save field data, filtered field data, and model input variables (field data linked to lidar metrics)
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(DENSversion))
final.to_csv('Input_variables_'+'v'+str(DENSversion)+'.csv',index=False)
df.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'.csv',index=False)
filt.to_csv('Field_data_'+str(watershed)+'_v'+str(DENSversion)+'_filtered.csv',index=False)

# Save RF model
joblib.dump(model_rf, 'RF_density_model_'+str(watershed)+'_v'+str(DENSversion)+'.joblib')

# Export normalization scalers
for n in range(len(all_scalers)):
    scaler=all_scalers[n]
    joblib.dump(scaler, 'scaler'+str(n+1)+'.pkl')

# Export error values
all_errors= pd.DataFrame({'rmse':[rmse_rf],'syst_error':[syst_error],'rand_error':[rand_error]})
all_errors.to_csv('model_error_values_v'+str(DENSversion)+'.csv')

# Save processing variables
var.to_csv(str(watershed)+'_ML_model_processing_variables_v'+str(DENSversion)+'.csv')

# Plot regressions between predictor variables and snow density (dependent variable)
# Calculate R-squared using scipy.stats.linregress
lst = ['year','phase','month','season','easting_m','northing_m','density_gcm3']
lst.extend(predictors)
final_long=final[lst]
final_long=final_long.melt(id_vars=['year','phase','month','season','easting_m','northing_m','density_gcm3'],
                      var_name='variable',value_name='value')
variables=final_long['variable'].unique()
R2s=[]
for n in (variables):
    x = final_long[final_long['variable']==n]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x["value"], x["density_gcm3"])
    r2 = r_value**2
    r2_text = f'$R^2$ = {r2:.3f}' # Format R-squared to 3 decimal places
    R2s.append(r2_text)
g = sns.lmplot(
    data=final_long,      # Your DataFrame
    x="value",            # X-axis variable
    y="density_gcm3",
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

