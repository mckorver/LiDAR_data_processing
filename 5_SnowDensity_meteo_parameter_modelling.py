# This script calculates Xt, a parameter used for snowdensity modelling
# Xt is calculated from air temperature (positive degree days) and snowfall measured at meteo stations
# Output:
# Raster maps of Xt, Snowfall, and positive degree days
# csv files of Xt, Snowfall, and positive degree days by elevation

# ACTION REQUIRED - ENTER REQUIREMENTS BELOW
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='CRU' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year='2023' # Enter year of interest
phases=['P1','P2','P3','P4','P5'] # Enter survey phases ('P1','P2', etc.)
BEversion = 2 # Enter Bare Earth version number.
resolution = 1 # Enter resolution in meters
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
rain_snow_threshold=0.90 # ENTER RAIN-SNOW THRESHOLD (according to Jennings et al., 2018: 0.97 for Metro Vancouver, 0.98 for Tsitika, 0.90 for Cruikshank, 0.91 for Englishman)

import os
import pandas as pd
import numpy as np
import numpy.core.defchararray as np_f
import matplotlib.pyplot as plt
import pyrsgis

# Import input data ------------------------------------------------------------------------------------------------
# Import bare earth and get min and max elev
[R,BE]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/Bare_earth/'+str(watershed)+'/DEM/v'+str(BEversion)+'/resolution_'+str(resolution)+'m/'+str(subbasin)+'_BE_v'+str(BEversion)+'_'+str(resolution)+'m.tif', bands='all'))
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
precip_pipe = np.array(pd.read_csv('WS_data_'+str(watershed)+'_'+str(year)+'.csv', usecols=['PC_low_mm']))
DateTime=np.array(pd.read_csv('WS_data_'+str(watershed)+'_'+str(year)+'.csv',usecols=['DateTime']))

# QAQC of total precipitation data -------------------------------------------------------------------
# Apply QAQC filters to total precip data
precip_pipe=precip_pipe.astype('float64')
#for n in range(1,len(precip_pipe)): # Remove pipe drains
#    x=precip_pipe[n]-precip_pipe[n-1]
#    if x<-10:
#        precip_pipe[n]=np.nan
#for n in range(1,len(precip_pipe)-1): # Remove erroneously high spikes
#    x=precip_pipe[n+1]-precip_pipe[n-1]
#    if x>20:
#        for m in range(-20,20):
#            precip_pipe[n+m]=np.nan
#for n in range(1,len(precip_pipe)): # Remove decreases (precip is cumulative so should never decrease)
#    x=precip_pipe[n]-precip_pipe[n-1]
#    if x<0:
#        precip_pipe[n]=precip_pipe[n-1]
#del n,x

df=pd.DataFrame(precip_pipe) 
#df2=np.array(df)

# Calculate hourly precipitation
#Precip=[]
#for n in range(1,len(df2)):
#    x=df2[n]-df2[n-1]
#    Precip.append(x)
Precip=np.array(df).flatten()
#Precip=np.insert(Precip,0,0)
#nans=np.argwhere(np.isnan(Precip))
#Precip[nans]=0
#Precip[Precip<0]=0
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
Xt_save.to_csv("Xt_1m_intervals_"+str(subbasin)+"_"+str(year)+".csv", index = False)
PDD_sum_save=pd.DataFrame(PDD_sum,columns=phases)
PDD_sum_save.insert(0, "Elevation",elevs)
PDD_sum_save.to_csv("PDD_1m_intervals_"+str(subbasin)+"_"+str(year)+".csv", index = False)
Total_snowfall_save=pd.DataFrame(Total_snowfall,columns=phases)
Total_snowfall_save.insert(0, "Elevation",elevs)
Total_snowfall_save.to_csv("S_1m_intervals_"+str(subbasin)+"_"+str(year)+".csv", index = False)

Xt_plot = Xt_save.plot(kind='line',x='Elevation',color=['red','blue','green'])
Xt_plot.set_ylabel('Xt')
Xt_plot.figure.savefig('Xt_1m_intervals_CRU_'+str(year)+'.png')
PDD_plot = PDD_sum_save.plot(kind='line',x='Elevation',color=['red','blue','green'])
PDD_plot.set_ylabel('PDD')
PDD_plot.figure.savefig('PDD_1m_intervals_CRU_'+str(year)+'.png')
Total_snowfall_plot = Total_snowfall_save.plot(kind='line',x='Elevation',color=['red','blue','green'])
Total_snowfall_plot.set_ylabel('Total snowfall')
Total_snowfall_plot.figure.savefig('Total_snowfall_1m_intervals_CRU_'+str(year)+'.png')

# Spatially distribute results according to elevation
# Distribute Xt
elev_data=np.around(BE,0)
Xt=np.transpose(Xt)
Xt=np.nan_to_num(Xt)
Xt_all=[]
for n in range(len(Xt)):
    xx=Xt[n]
    yy=np.copy(elev_data)
    for m in range(len(elevs)):
        x=elevs[m]
        y=xx[m]
        yy[yy==x]=y
    Xt_all.append(yy)
    print(n)
del n,x,y,xx,yy,m,Xt,elev_data

# Export spatially-distributed Xt
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution)+'m/')
for n in range(len(Xt_all)):
    pyrsgis.raster.export(Xt_all[n], R, filename='Distributed_Xt_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif')

# Distribute PDD
elev_data=np.around(BE,0)
PDD=np.transpose(PDD_sum)
PDD=np.nan_to_num(PDD)
PDD_all=[]
for n in range(len(PDD)):
    xx=PDD[n]
    yy=np.copy(elev_data)
    for m in range(len(elevs)):
        x=elevs[m]
        y=xx[m]
        yy[yy==x]=y
    PDD_all.append(yy)
    print(n)
del n,x,y,xx,yy,m,PDD,elev_data,min_elev,max_elev

# Export spatially-distributed PDD
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution)+'m/')
for n in range(len(phases)):
    pyrsgis.raster.export(PDD_all[n], R, filename='Distributed_PDD_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif')
    
# Distribute snowfall
elev_data=np.around(BE,0)
S=np.transpose(Total_snowfall)
S=np.nan_to_num(S)
S_all=[]
for n in range(len(S)):
    xx=S[n]
    yy=np.copy(elev_data)
    for m in range(len(elevs)):
        x=elevs[m]
        y=xx[m]
        yy[yy==x]=y
    S_all.append(yy)
    print(n)
del n,x,y,xx,yy,m,S,elev_data

# Export spatially-distributed snowfall
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Meteorological_parameter_modelling/'+str(year)+'/Output/resolution_'+str(resolution)+'m/')
for n in range(len(S_all)):
    pyrsgis.raster.export(S_all[n], R, filename='Distributed_Snowfall_'+str(subbasin)+'_'+str(year)+'_'+str(phases[n])+'.tif')
