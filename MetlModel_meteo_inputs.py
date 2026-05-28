# This script calculates Xt, a parameter used for snowdensity modelling
# Xt is calculated from air temperature (positive degree days) and snowfall measured at meteo stations
# Output:
# Raster maps of Xt, Snowfall, and positive degree days
# csv files of Xt, Snowfall, and positive degree days by elevation

import os
import pandas as pd
import numpy as np
import numpy.core.defchararray as np_f
import matplotlib.pyplot as plt
import pyrsgis

# Import input data ------------------------------------------------------------------------------------------------
# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
BEversion = var['BEversion'][0]
phases = []
x = var['phases'][var['phases'].notna()]
for n in range(len(x)):
    a = x[n]
    phases.append(a)

# RAIN-SNOW THRESHOLD (according to Jennings et al., 2018)
if watershed == 'TSI':
    rain_snow_threshold=0.98 
elif watershed == 'CRU':
    rain_snow_threshold=0.90
elif watershed == 'ENG':
    rain_snow_threshold=0.91
elif watershed == 'MV':
    rain_snow_threshold=0.97
    
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

# Import required metadata
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year))
survey_dates=[]
for n in phases:
    x=pd.read_csv('Metadata_'+str(watershed)+'_'+str(year)+'.csv')
    y=x[x['survey']==n]
    y=y['survey_days'].iloc[0]
    survey_dates.append(y)
WSup_elev=int(np.array(pd.read_csv('Metadata_'+str(watershed)+'_'+str(year)+'.csv', usecols=['WSup_elev']))[0])
WSlow_elev=int(np.array(pd.read_csv('Metadata_'+str(watershed)+'_'+str(year)+'.csv', usecols=['WSlow_elev']))[0])

# Import weather station data
Tair_up = np.array(pd.read_csv('WS_data_'+str(watershed)+'_'+str(year)+'.csv', usecols=['Tair_up']))
Tair_low = np.array(pd.read_csv('WS_data_'+str(watershed)+'_'+str(year)+'.csv', usecols=['Tair_low']))
precip_pipe = np.array(pd.read_csv('WS_data_'+str(watershed)+'_'+str(year)+'.csv', usecols=['PC_up_mm']))
DateTime=np.array(pd.read_csv('WS_data_'+str(watershed)+'_'+str(year)+'.csv',usecols=['DateTime']))

# Reformat and check total precip data
precip_pipe=precip_pipe.astype('float64')
df=pd.DataFrame(precip_pipe) 
Precip=np.array(df).flatten()
plt.plot(Precip)
plt.show() 
del x,n,df,precip_pipe

# Calculations -----------------------------------------------------------------------------
# Calculate atmospheric lapse rate
lapse_rate = (Tair_up-Tair_low)/(WSup_elev-WSlow_elev)
del WSlow_elev,Tair_low

# Correct air temperature for specified atmospheric lapse rate
elevs = np.arange(start=min_elev, stop=max_elev, step=1) 
temp_diffs=[]
for n in range(len(elevs)):
    x=(elevs[n]-WSup_elev)*lapse_rate
    temp_diffs.append(x)
Corrected=[]
Tair_corrected=[]
for n in range(len(temp_diffs)):
    Corrected=Tair_up+temp_diffs[n]
    Tair_corrected.append(Corrected)

# Calculate PDH (positive degree hour) sum
PDH_all=[]
for n in range(len(temp_diffs)):
    non_PDH=np.where(Tair_corrected[n]<0)
    PDH=Tair_corrected[n]
    PDH[non_PDH] = 0
    PDH_all.append(PDH)

# Calculate snowfall
Corrected1=[]
Tair_corrected1=[]
for n in range(len(temp_diffs)):
    Corrected1=Tair_up+temp_diffs[n]
    Corrected1=np.reshape(Corrected1,(len(Corrected1),))
    Tair_corrected1.append(Corrected1)
Snowfall=[]
snow_rain1=Tair_corrected1
snow_rain = snow_rain1
for l in range(len(temp_diffs)):
    non_snow=np.where(snow_rain[l]>rain_snow_threshold)
    snow=np.where(snow_rain[l]<=rain_snow_threshold)
    snow_vs_rain=snow_rain[l]
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
    PDD_daily_test=[sum(x)/len(x) for x in sorted_PDH_all[z]]
    PDD_daily_allsites.append(PDD_daily_test)

# Find total daily snowfall for elevation band
snowfall_daily_allsites=[]
for z in range(len(temp_diffs)):
    snowfall_daily_test=[sum(x) for x in sorted_snowfall_all[z]]
    snowfall_daily_allsites.append(snowfall_daily_test)

# Output ---------------------------------------------------------------------------------------------------
PDD_daily_elev=np.array(PDD_daily_allsites)
PDD_daily_elev=np.transpose(PDD_daily_elev)
Daily_snowfall=np.array(snowfall_daily_allsites)
Daily_snowfall=Daily_snowfall/10
Daily_snowfall=np.transpose(Daily_snowfall)

os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Future_SWE/INPUTS')
PDD_sum_save=pd.DataFrame(PDD_daily_elev,columns=elevs)
PDD_sum_save.to_csv("PDD_daily_allsites_"+str(year)+".csv", index = False)
Total_snowfall_save=pd.DataFrame(Daily_snowfall,columns=elevs)
Total_snowfall_save.to_csv("SNOW_daily_allsites_"+str(year)+".csv", index = False)
