
# ACTION REQUIRED BELOW
watershed='TSI' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='SC' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2025' # Enter year of interest
phases=['D1','D2','D3','D4','D5','D6'] # Enter survey phases ('P1','P2', etc.)
BEversion=1 # Enter Bare Earth version number.
resolution=1 # Enter resolution in meters
drive = 'E'
lidar = 'RPAS' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone

# Import packages
import rasterio
import os
import numpy as np
import pandas as pd

# Read gridded input datasets       
os.chdir(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/')     
slope = rasterio.open(str(subbasin)+'_Slope_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
aspect = rasterio.open(str(subbasin)+'_Aspect_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
curvature = rasterio.open(str(subbasin)+'_Curvature_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
canopy_cover = rasterio.open(str(subbasin)+'_CC_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')
canopy_height = rasterio.open(str(subbasin)+'_CH_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif')

s_slope = []
s_aspect = []
s_curvature = []
s_canopyc = []
s_canopyh = []
s_Xt = []
s_PDD = []
s_Snow = []
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
