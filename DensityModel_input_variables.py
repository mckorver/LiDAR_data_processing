
# ACTION REQUIRED BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='CRU' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
years=['2021','2022','2023','2024','2025'] # Enter year of interest
phases=[['P1','P2','P3','P4','P5'],['P1','P2','P3','P4'],['P1','P2','P3','P4','P5'],['P1'],['P1','P2','P3']] # Enter survey phases ('P1','P2', etc.) for each year
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

# Import field data
datetimes_field=[]
datetimes_aco=[]
eastings=[]
northings=[]
depths=[]
cores=[]
densities=[]
manual_remove=[]
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
        datetimes_field.append(datetime_f)
        datetimes_aco.append(datetime_l)
        eastings.append(easting)
        northings.append(northing)
        depths.append(depth)
        cores.append(core)
        densities.append(density)
        manual_remove.append(manual_rem)

df=pd.DataFrame({"datetime_field":np.concatenate(datetimes_field),"datetime_aco":np.concatenate(datetimes_aco),
                 "easting_m":np.concatenate(eastings),"northing_m":np.concatenate(northings),
                 "snow_depth":np.concatenate(depths),"core_depth":np.concatenate(cores),
                 "density":np.concatenate(densities), "manual_remove":np.concatenate(manual_remove)})
df['time_gap_hr']=df['datetime_field']-df['datetime_aco']
df['time_gap_hr'] = df['time_gap_hr'].dt.total_seconds()/3600
df['time_gap_hr'] = df['time_gap_hr'].abs().astype('int64')
df['year'] = df['datetime_field'].dt.year
df['month'] = df['datetime_field'].dt.month

# QAQC field data
df=df[(df['density'].notna())&(df['density']>0)]
df['flag'] = 'AV'
df.loc[(df['time_gap_hr']>60), 'flag'] = 'time'
df.loc[(df['manual_remove']=='Y'), 'flag'] = 'manual'
df.loc[(df['density']>0.8)|(df['density']<0.1), 'flag'] = 'range'
df.loc[(df['snow_depth']-df['core_depth']<-5)|(df['snow_depth']/df['core_depth']>=2), 'flag'] = 'core'
filt=df[(df['flag']=='AV')]

os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/ML_density_model/v2')
df.to_csv('Input_variables_'+str(watershed)+'_v2.csv',index=False)

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
canopy_height = rasterio.open(str(subbasin)+'_CH_v'+str(BEversion)+'_'+str(resolution)+'m.tif')

# Import modelled meteo parameters
Xt = []
PDD = []
Snow = []
for a in range(len(years)):
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(years[a])+'/Output/resolution_'+str(resolution)+'m/')
    for b in range(len(phases[a])):
        # Read gridded meteo parameters
        xt = rasterio.open('Distributed_Xt_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.tif')
        p = rasterio.open('Distributed_PDD_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.tif')
        s = rasterio.open('Distributed_Snowfall_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][n])+'.tif')


s_slope = []
s_aspect = []
s_curvature = []
s_canopyc = []
s_canopyh = []
eastings=[]
northings=[]
elevations=[]
depths=[]
densities=[]
for n in range(len(phases)):
    # Read gridded meteo parameters
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output')
    xt = rasterio.open('Distributed_Xt_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif')
    p = rasterio.open('Distributed_PDD_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif')
    s = rasterio.open('Distributed_Snowfall_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif')

    # Read field data collection coordinates
    os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/'+str(year))
    easting=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'.csv', usecols=['easting'])).astype('float64')
    easting=easting.reshape(len(easting),)
    northing=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'.csv', usecols=['northing'])).astype('float64')
    northing=northing.reshape(len(northing),)
    elevation=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'.csv', usecols=['Ellipsoid_Height_m'])).astype('float64')
    elevation=elevation.reshape(len(elevation),)
    depth=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'.csv', usecols=['snow_depth'])).astype('float64')
    depth=depth.reshape(len(depth),)
    density=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(year)+'_'+str(phases[n])+'.csv', usecols=['density'])).astype('float64')
    density=density.reshape(len(density),)

    # Sample gridded datasets at field data collection points
    c=[]
    for m in range(len(easting)):
        cl=(easting[m],northing[m])
        c.append(cl)
    s_s = [x for x in slope.sample(c)]
    s_a = [x for x in aspect.sample(c)]
    s_c = [x for x in curvature.sample(c)]
    s_cc = [x for x in canopy_cover.sample(c)]
    s_ch = [x for x in canopy_height.sample(c)]
    s_X = [x for x in xt.sample(c)]
    s_P = [x for x in p.sample(c)]
    s_Sn = [x for x in s.sample(c)]
    s_Xt.append(s_X)
    s_PDD.append(s_P)
    s_Snow.append(s_Sn)
    s_slope.append(s_s)
    s_aspect.append(s_a)
    s_curvature.append(s_c)
    s_canopyc.append(s_cc)
    s_canopyh.append(s_ch)
    eastings.append(easting)
    northings.append(northing)
    elevations.append(elevation)
    depths.append(depth)
    densities.append(density)

sampled_slope = []
sampled_aspect = []
sampled_curvature = []
sampled_canopycover = []
sampled_canopyheight = []
sampled_Xt = []
sampled_PDD = []
sampled_Snow = []
for n in range(len(phases)):
    ss_Xt=(np.array(s_Xt[n]).astype('float64')).reshape(len(s_Xt[n]),)
    nans=np.where(ss_Xt<0)
    ss_Xt[nans]='NaN'
    ss_PDD=(np.array(s_PDD[n]).astype('float64')).reshape(len(s_PDD[n]),)
    nans=np.where(ss_PDD<0)
    ss_PDD[nans]='NaN'
    ss_Snow=(np.array(s_Snow[n]).astype('float64')).reshape(len(s_Snow[n]),)
    nans=np.where(ss_Snow<0)
    ss_Snow[nans]='NaN'
    ss_aspect=(np.array(s_aspect[n]).astype('float64')).reshape(len(s_aspect[n]),)
    ss_slope=(np.array(s_slope[n]).astype('float64')).reshape(len(s_slope[n]),)
    nans=np.where(ss_slope<0)
    ss_slope[nans]='NaN'
    ss_curvature=(np.array(s_curvature[n]).astype('float64')).reshape(len(s_curvature[n]),)
    nans=np.where(ss_curvature<0)
    ss_curvature[nans]='NaN'
    ss_canopyc=(np.array(s_canopyc[n]).astype('float64')).reshape(len(s_canopyc[n]),)
    nans=np.where(ss_canopyc<0)
    ss_canopyc[nans]=0
    ss_canopyh=(np.array(s_canopyh[n]).astype('float64')).reshape(len(s_canopyh[n]),)
    nans=np.where(ss_canopyh<0)
    highs=np.where(ss_canopyh>100)
    ss_canopyh[nans]=0
    ss_canopyh[highs]=0
    sampled_Xt.append(ss_Xt)
    sampled_PDD.append(ss_PDD)
    sampled_Snow.append(ss_Snow)
    sampled_slope.append(ss_slope)
    sampled_aspect.append(ss_aspect)
    sampled_curvature.append(ss_curvature)
    sampled_canopycover.append(ss_canopyc)
    sampled_canopyheight.append(ss_canopyh)

all_sampled_data=[] 
# Merge data for each survey
for n in range(len(phases)):
    all=pd.DataFrame({"easting":eastings[n],"northing":northings[n],"elevation":elevations[n],"depth":depths[n],"density":densities[n],"slope":sampled_slope[n],"aspect":sampled_aspect[n],"curvature":sampled_curvature[n],"Xt":sampled_Xt[n],"PDD_sum":sampled_PDD[n],"Total_snowfall":sampled_Snow[n],"canopy_cover":sampled_canopycover[n],"canopy_height":sampled_canopyheight[n]})
    all_sampled_data.append(all)

# Export sampled data
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Plot_characteristic_sampling/'+str(year))
for n in range(len(phases)):
    all_sampled_data[n].to_csv('All_plot_characteristics_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.csv')
