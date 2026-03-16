
# ACTION REQUIRED BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
years=['2021','2022','2023','2024','2025'] # Enter year of interest
phases=[['P1','P2','P3','P4','P5'],['P1','P2','P3','P4'],['P1','P2','P3','P4','P5'],['P1'],['P1','P2','P3']] # Enter survey phases ('P1','P2', etc.) for each year
modelversion=2
BEversion=2 # Enter Bare Earth version number.
CANversion = 2 # Enter Canopy version number
resolution=1 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone

# Import packages
import rasterio
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(modelversion))
df.to_csv('Field_data_'+str(watershed)+'_v2.csv',index=False)
filt.to_csv('Field_data_'+str(watershed)+'_v2_filtered.csv',index=False)
del datetimes_field,datetimes_aco,eastings,northings,depths,cores,densities,manual_remove,field_phases

## Import LiDAR derived input variables and sample for field locations
# Import Bare Earth metrics    
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/')     
elevations = rasterio.open(str(watershed)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
slope = rasterio.open(str(watershed)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
curvature = rasterio.open(str(watershed)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
aspect = rasterio.open(str(watershed)+'_Aspect_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
northness = rasterio.open(str(watershed)+'_Northness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
eastness = rasterio.open(str(watershed)+'_Eastness_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')

# Import canopy metrics
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/Canopy/v'+str(CANversion)+'/resolution_'+str(resolution)+'m/')    
canopy_density = rasterio.open(str(watershed)+'_CD_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
canopy_cover = rasterio.open(str(watershed)+'_CC_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
canopy_height = rasterio.open(str(watershed)+'_CH_v'+str(BEversion)+'_'+str(resolution)+'m.tif')

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
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(years[a])+'/Output/resolution_'+str(resolution)+'m/')
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

final_long=final.melt(id_vars=['year','phase','month','season','easting_m','northing_m','density_gcm3'],
                      var_name='variable',value_name='value')

# Calculate R-squared using scipy.stats.linregress
# The function returns slope, intercept, r_value, p_value, std_err
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
    hue="season",     # Y-axis variable
    col="variable",       # Variable for columns (facets)
    col_wrap=4,
    sharex=False,                
    height=4,             # Height of each facet in inches
    aspect=1.2,           # Aspect ratio of each facet
    line_kws={'color': 'red'}) # Keyword arguments for the regression line
for n in range(len(variables)):
    ax=g.axes[n]
    ax.text(0.05, 0.9, R2s[n], transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

# Export sampled data
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v'+str(modelversion))
final.to_csv('Input_variables_'+'_v'+str(modelversion)+'.csv',index=False)
