# This script calculates Xt, a parameter used for snowdensity modelling
# Xt is calculated from air temperature (positive degree days) and snowfall measured at meteo stations
# Output:
# Raster maps of Xt, Snowfall, and positive degree days
# csv files of Xt, Snowfall, and positive degree days by elevation

import os
import pandas as pd
import numpy as np
import pyrsgis
from datetime import datetime
from pathlib import Path

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

# Import existing Snowfall, AirTemp(>=0), and Xt data
os.chdir(str(drive)+':/LiDAR_data_processing/Weather_station_data/'+str(watershed)+'/')
file=Path('Xt_daily_elevation.csv')
if file.is_file():
    Xt_archived=pd.read_csv('Xt_daily_elevation.csv')
    Snowfall_archived=pd.read_csv('Snowfall_daily_elevation.csv')
    AirTemp_archived=pd.read_csv('AirTemp_daily_elevation.csv')

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

# Import station metadata
metadata = pd.read_csv('metadata/Metadata_'+str(watershed)+'.csv')
WSup_airT=metadata['WSup_airT'][0]
WSlow_airT=metadata['WSlow_airT'][0]
WS_precip=metadata['WS_precip'][0]
WSup_airT_elev=metadata['WSup_airT_elev'][0]
WSlow_airT_elev=metadata['WSlow_airT_elev'][0]
WS_precip_elev=metadata['WS_precip_elev'][0]
calc_start=datetime.strptime(metadata['calc_start'][0],"%Y-%m-%d %H:%M")
calc_end=datetime.strptime(metadata['calc_end'][0],"%Y-%m-%d %H:%M")

# Import weather station data
Tair_up = pd.read_csv(WSup_airT+'_airtemp_qaqc.csv')
Tair_up['Tair_up']=Tair_up['observed'].fillna(Tair_up['predicted'])
Tair_up['datetime']=Tair_up['datetime'].astype('datetime64[ns]')
Tair_low = pd.read_csv(WSlow_airT+'_airtemp_qaqc.csv')
Tair_low['Tair_low']=Tair_low['observed'].fillna(Tair_low['predicted'])
Tair_low['datetime']=Tair_low['datetime'].astype('datetime64[ns]')
Precip = pd.read_csv(WS_precip+'_precip_qaqc.csv')
Precip['precip'] = Precip['precip_qaqc'].diff()
Precip.loc[(Precip['precip']<0)|(Precip['precip'].isna()),'precip'] = 0
Precip['datetime']=Precip['datetime'].astype('datetime64[ns]')

Wx=pd.merge(pd.merge(Tair_up[['datetime','Tair_up']],Tair_low[['datetime','Tair_low']], on ='datetime', how='inner'),
            Precip[['datetime','precip']], on ='datetime', how='inner')
Wx['date']=Wx['datetime'].dt.strftime('%Y-%m-%d')
del(Tair_up,Tair_low,Precip)

# Calculations -----------------------------------------------------------------------------
# Calculate air temperature for each elevation and each timestamp based on atmospheric lapse rate
lapse_rate = np.array((Wx['Tair_up']-Wx['Tair_low'])/(WSup_airT_elev-WSlow_airT_elev))
elevs = np.arange(start=min_elev, stop=max_elev+1, step=1) 
Tair_array=[]
for n in range(len(elevs)):
    x=np.array(((elevs[n]-WSup_airT_elev)*lapse_rate)+np.array(Wx['Tair_up']))
    Tair_array.append(x)

# Calculate snowfall for each elevation and each timestamp
Snow_array=[]
precip=np.array(Wx['precip'])
for n in range(len(elevs)):
    snow_vs_rain=Tair_array[n].copy()
    snow_vs_rain=np.reshape(snow_vs_rain,(len(snow_vs_rain),))
    non_snow=np.where(snow_vs_rain>rain_snow_threshold)
    snow=np.where(snow_vs_rain<=rain_snow_threshold)
    snow_vs_rain[non_snow]=0
    snow_vs_rain[snow]=1
    snowfall=np.multiply(snow_vs_rain,precip)
    Snow_array.append(snowfall)

Tair=Wx[['date']]
Snow=Wx[['date']]
for n in range(len(elevs)):
    Tair[str(elevs[n])]=Tair_array[n]
    Snow[str(elevs[n])]=Snow_array[n]

# Find mean daily Air Temps (set negative mean Air Temps to 0) and total daily snowfall for each elevation band
numeric_cols = Tair.select_dtypes(include='number').columns
#Tair[numeric_cols] = Tair[numeric_cols].mask(Tair[numeric_cols] < 0, 0) # Option 1: Set to 0 BEFORE averaging
AirT_daily=Tair.groupby('date',as_index=False).mean()
x = AirT_daily.loc[:,AirT_daily.columns != 'date']
x[x<0]=0 # Option 2: Set to 0 after averaging
x.insert(0,'date',AirT_daily['date'])
AirT_daily=x
AirT_daily['date']=pd.to_datetime(AirT_daily['date'])
Snowfall_daily=Snow.groupby('date',as_index=False).sum()
Snowfall_daily['date']=pd.to_datetime(Snowfall_daily['date'])

# Calculate Xt for each date and each elevation, calculated from preceding Sept 1st.
dates=Snowfall_daily[['date']]
dates=dates[dates['date'].between(calc_start, calc_end)]
dates['date']=pd.to_datetime(dates['date'])
for n in range(len(dates)):
    date=dates.iloc[n,0]
    if date.month<9: 
        date_start=str(date.year-1) + "-09-01"
    else:
        date_start=str(date.year) + "-09-01"
    date_end=date.strftime('%Y-%m-%d')
    PDD_ex=AirT_daily[AirT_daily['date'].between(date_start, date_end)].copy() # Extract daily Air Temp data between preceding Sep 1st and date 'n'
    PDD_ex=PDD_ex.drop('date',axis='columns')
    PDD_rev=pd.DataFrame()
    for m in elevs:
        PDD_rev[str(m)]=PDD_ex.loc[::-1,str(m)].cumsum()[::-1] # Calculate reverse cumulative PDD from daily Air Temp for each elevation band  
    Snowfall_ex=Snowfall_daily[Snowfall_daily['date'].between(date_start, date_end)].copy() # Extract daily snowfall data between preceding Sep 1st and field date
    Snowfall_ex=Snowfall_ex.drop('date',axis='columns')
    # Option 1: sum BEFORE dividing
    #a1=PDD_rev.sum(numeric_only=True).to_frame().T
    #a2=Snowfall_ex.sum(numeric_only=True).to_frame().T
    #b1=a2.div(a1).replace(np.inf, np.nan).fillna(0) # snowfall / reverse cumulative PDD
    # Option 2: sum AFTER dividing
    a=Snowfall_ex.div(PDD_rev).replace(np.inf, np.nan).fillna(0) # snowfall / reverse cumulative PDD
    b1=a.sum(numeric_only=True).to_frame().T # Sum of (snowfall / reverse cumulative PDD)
    b2=b1.div(b1).replace(np.inf, np.nan).fillna(0) # create dataframe of ones
    b=b2.div(b1).replace(np.inf, np.nan).fillna(0) # # 1/(Sum of (snowfall / reverse cumulative PDD))
    if n==0:
        Xt=b.copy()
    else:
        Xt=pd.concat([Xt,b],ignore_index=True)
Xt.insert(0,'date',dates)

# Update the existing data file with the new data
if file.is_file():
    Xt_new=Xt_archived.copy()
    Xt_new.set_index('date',inplace=True)
    Xt.set_index('date',inplace=True)
    Xt_new.update(Xt)
    Xt_new.reset_index(inplace=True)
    Xt.reset_index(inplace=True)
    last_day_archived=datetime.strptime(Xt_new['date'].iloc[-1],"%Y-%m-%d")
    Xt=Xt[Xt['date'] > last_day_archived]
    Xt['date']=Xt['date'].dt.strftime('%Y-%m-%d')
    Xt_new=pd.concat([Xt_new,Xt],ignore_index=True)
    Xt_archived.to_csv('Xt_daily_elevation_archived.csv',index=False)
    Xt_new.to_csv('Xt_daily_elevation.csv',index=False)
    AirT_daily.to_csv('AirTemp_daily_elevation.csv',index=False)
    Snowfall_daily.to_csv('Snowfall_daily_elevation.csv',index=False)
else:
    Xt.to_csv('Xt_daily_elevation_Maartjemethod.csv',index=False)
    AirT_daily.to_csv('AirTemp_daily_elevation_Maartjemethod.csv',index=False)
    Snowfall_daily.to_csv('Snowfall_daily_elevation_Maartjemethod.csv',index=False)



















