# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

# 'CRU','Comox','Eric','Moat','Rees','Residual'
# 'MV','SEY','CAP','BurwellLake','LochLomond','PalisadeLake','UpperSeymour'
# 'ENG','Arrowsmith','Fishtail','Cokely'

# ACTION REQUIRED BELOW
# To run this code, use a conda environment configured for rasterio
watershed='CRU' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin= ['CRU','Comox','Eric','Moat','Rees','Residual'] #Enter prefixes for all subbasins, including the watershed.
year='2025' # ENTER YEAR OF INTEREST
phases=['P1','P2','P3'] # Enter survey numbers ,'P2','P3','P4','P5' NOTE run all surveys of a year simultaneously
resolution=2
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone
date='20251016' #Enter date of today
lakemodel = 'N' # Enter 'Y' or 'N' for including modelled SnowDepth on lakes
glaciermodel = 'N' # Enter 'Y' or 'N' for including a SWE model for glaciers

import pandas as pd
import glob
import os
import numpy as np
import pyrsgis
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate watershed and subbasin areas
WS_areas=[]
for a in range(len(subbasin)):
    # Import watershed mask (without lakes or glaciers)
    if lakemodel == 'Y' and glaciermodel == 'Y':
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin[a])+'_watershed_'+str(resolution)+'m.tif'))
    elif lakemodel == 'N' and glaciermodel == 'Y':
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin[a])+'_watershed_no_lakes_'+str(resolution)+'m.tif'))
    elif lakemodel == 'N' and glaciermodel == 'N':
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution)+'m/'+str(subbasin[a])+'_watershed_no_lakes_no_glaciers_'+str(resolution)+'m.tif'))
    nans=np.where(WS<1)
    WS[nans]=np.nan
    area=np.nansum(WS)*resolution*resolution
    WS_areas.append(area)
WS_areas=pd.DataFrame(list(zip(subbasin,WS_areas)),columns=['watershed','WS_area_m2'])
WS_areas.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Tables/'+str(watershed)+'/'+str(year)+'/'+str(date)+'/'+str(watershed)+'_WS_areas_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.csv',index=False)

# Create a summary table
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Key_numbers/'+str(year)+'/resolution_'+str(resolution)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/')
df1=pd.DataFrame()
for a in range(len(subbasin)):
    # Create a summary table
    files1=glob.glob(str(subbasin[a])+'**.csv')
    merged=pd.DataFrame()
    for n in files1:
        df = pd.read_csv(n).set_index('Survey')
        last_col = df.columns[-1]
        merged = merged.join(df[last_col], how='outer')
    if subbasin[a] != watershed:
        merged['Absolute_total_SWV_errors_m3']=0
    x=merged['Absolute_total_SWV_errors_m3']
    for n in range(len(phases)):
        if x[n]==0:
            x[n]=merged['Percentage_total_SWV_errors'][n]*merged['Total_SWV_m3'][n]/100
        else:
            x[n]=x[n]
    merged['Absolute_total_SWV_errors_m3']=x
    merged['Absolute_mean_SWE_error']=(merged['Percentage_total_SWV_errors']*merged['Mean_SWE_mm']/100)
    merged=merged.assign(watershed=str(subbasin[a]))
    merged=merged.assign(year=year)
    merged['Mean_snow_depth_m'] = merged['Mean_snow_depth_m'].round(2)
    merged['Mean_snow_depth_aboveSL_m'] = merged['Mean_snow_depth_aboveSL_m'].round(2)
    merged['Mean_snow_density_kgm3'] = merged['Mean_snow_density_kgm3'].round(0)
    merged['Total_snow_m3'] = merged['Total_snow_m3'].round(0)
    merged['Total_SWV_m3'] = merged['Total_SWV_m3'].round(0)
    merged['Mean_SWE_mm'] = merged['Mean_SWE_mm'].round(0)
    merged['Absolute_mean_SWE_error'] = merged['Absolute_mean_SWE_error'].round(0)
    merged['Absolute_total_SWV_errors_m3'] = merged['Absolute_total_SWV_errors_m3'].round(0)
    merged['Percentage_total_SWV_errors'] = merged['Percentage_total_SWV_errors'].round(2)
    df1=df1.append(merged)
df1 = df1.reset_index()
cols = df1.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols = cols[-1:] + cols[:-1]
sum_table = df1[cols]
sum_table.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Tables/'+str(watershed)+'/'+str(year)+'/'+str(date)+'/Summary_table_'+str(year)+'_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.csv',index=False)
del df1,cols

# Create an elevation banded summary table
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Tables/')
df1 = pd.read_csv('Survey_dates_for_plotting.csv')
df1['year'] = pd.to_numeric(df1['year'])
df1 = df1[df1['watershed']==str(watershed)]
df1 = df1[df1['year']==pd.to_numeric(year)]
df1 = df1.drop(columns='watershed')
for a in range(len(subbasin)): 
    all_merged=[]
    for n in range(len(phases)):
        file1=str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_Elevation_banded_mean_SWE.csv'
        file2=str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_Elevation_banded_total_SWV.csv'
        result1=pd.DataFrame(pd.read_csv(file1)).set_index('elev_band')
        result2=pd.DataFrame(pd.read_csv(file2)).set_index('elev_band')
        last_col2 = result2.columns[-1]
        merged=result1.join(result2[last_col2], how='outer')
        merged=merged.drop('Unnamed: 0', axis=1)
        merged['SWE_mean_mm'] = merged['SWE_mean_mm'].round(0)
        merged['Total_SWV_m3'] = merged['Total_SWV_m3'].round(0)
        all_merged.append(merged)
    for n in range(len(phases)):
        x=all_merged[n]
        y = df1[df1['Survey']==str(phases[n])]
        x['watershed'] = subbasin[a]
        x['date'] = y.iloc[0]['date']
        x.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Tables/'+str(watershed)+'/'+str(year)+'/'+str(date)+'/Elevation_tables/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/Elevation_table_'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'.csv')

# Update the 'Yearly_comparison' csv
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Tables/')
df1 = pd.read_csv(str(watershed)+'/Yearly_comparison.csv')
df1 = df1[df1["year"] != int(year)]
df1 = df1[['watershed','year','Survey','date','date_figure','Total_SWV_m3','Absolute_total_SWV_errors_m3','Mean_SWE_mm','Absolute_mean_SWE_error']]
df2 = pd.read_csv('Survey_dates_for_plotting.csv')
df2 = df2[df2["watershed"] == str(watershed)]
df2 = df2[df2["year"] == int(year)]
df2 = df2.drop(['watershed','year'], axis = 1)
df3 = sum_table[['Survey','year','watershed','Total_SWV_m3','Absolute_total_SWV_errors_m3','Mean_SWE_mm','Absolute_mean_SWE_error']]
df4 = pd.merge(df2,df3,how='right',on=['Survey'])
df4 = df4[['watershed','year','Survey','date','date_figure','Total_SWV_m3','Absolute_total_SWV_errors_m3','Mean_SWE_mm','Absolute_mean_SWE_error']]

df = pd.concat([df1,df4])
df.to_csv(str(watershed)+'/Yearly_comparison.csv',index=False)
del df1,df2,df3,df4

### PLOTS -------------------------------------------------------------------------





### RUN THIS SECTION WHEN ALL YEARS WERE RUN
# Create Table for Yearly Comparison Figure
yearly=pd.DataFrame()
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Tables/')
df1 = pd.read_csv('Survey_dates_for_plotting.csv')
df1 = df1[df1['watershed']==str(watershed)].set_index(['Survey','year'])
df1 = df1.drop(columns='watershed')
files1=glob.glob(str(watershed)+'/'+str(date)+'/Summary_table_**.csv')
df2=pd.DataFrame()
for n in files1:
    x = pd.read_csv(n)
    df2=df2.append(x)
df2 = df2.set_index(['Survey','year'])
yearly = df1.join(df2[['watershed','Total_SWV_m3','Absolute_total_SWV_errors_m3','Mean_SWE_mm','Absolute_mean_SWE_error']], how='outer')
yearly.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/Tables/'+str(watershed)+'/'+str(date)+'/Yearly_comparison_figure.csv')
