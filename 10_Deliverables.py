# NOTE this script has to be run on all surveys of a year, results for all surveys are outputted into one file

import pandas as pd
import glob
import os
import numpy as np
import pyrsgis
import matplotlib.pyplot as plt
import seaborn as sns

# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/ACO/input_data/Processing_variables.csv', dtype={'year':str, 'resolution1':str, 'resolution2':str,'BEversion':str, 'CANversion':str, 'date':str})
watershed = var['watershed'][0]
extent = var['extent'][0]
year = var['year'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution2 = var['resolution2'][0]
BEversion = var['BEversion'][0]
glaciers = var['glaciers'][0]
glaciermodel = var['glaciermodel'][0]
lakemodel = var['lakemodel'][0]
date = var['date'][0]
phases = []
subbasin = []
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if isinstance(y, float):
            a.append(int(y))
        else:
            a.append(y)
append_fun(phases,'phases')
append_fun(subbasin,'subbasin')

# Create new folder named as the 'date' variable
path = str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Figures_tables/'+str(date)
os.makedirs(path, exist_ok=True) 

# Calculate watershed and subbasin areas
WS_areas=[]
for a in range(len(subbasin)):
    # Import watershed mask (without lakes or glaciers)
    if lakemodel == 'Y' and (glaciermodel == 'Y' or glaciers =='N'):
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution2)+'m/'+str(subbasin[a])+'_watershed_'+str(resolution2)+'m.tif'))
    elif lakemodel == 'N' and (glaciermodel == 'Y' or glaciers =='N'):
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution2)+'m/'+str(subbasin[a])+'_watershed_no_lakes_'+str(resolution2)+'m.tif'))
    elif lakemodel == 'N' and glaciermodel == 'N':
        [R,WS]=np.array(pyrsgis.raster.read(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Snow_depth_processing/'+str(watershed)+'/watershed_mask/resolution_'+str(resolution2)+'m/'+str(subbasin[a])+'_watershed_no_lakes_no_glaciers_'+str(resolution2)+'m.tif'))
    nans=np.where(WS<1)
    WS[nans]=np.nan
    area=np.nansum(WS)*int(resolution2)*int(resolution2)
    WS_areas.append(area)
WS_areas=pd.DataFrame(list(zip(subbasin,WS_areas)),columns=['subbasin','WS_area_m2'])
if glaciers == 'Y':
    WS_areas.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Figures_tables/'+str(date)+'/'+str(watershed)+'_WS_areas_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.csv',index=False)
else:
    WS_areas.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Figures_tables/'+str(date)+'/'+str(watershed)+'_WS_areas_lakemodel'+str(lakemodel)+'.csv',index=False)

# Create a summary table
if glaciers == 'Y':
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Key_numbers/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/')
else:
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Key_numbers/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'/')
df1=pd.DataFrame()
for a in range(len(subbasin)):
    # Create a summary table
    files1=glob.glob(str(subbasin[a])+'**.csv')
    merged=pd.DataFrame()
    for n in files1:
        df = pd.read_csv(n).set_index('Survey')
        last_col = df.columns[-1]
        merged = merged.join(df[last_col], how='outer')
    if subbasin[a] != extent:
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
if glaciers == 'Y':
    sum_table.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Figures_tables/'+str(date)+'/Summary_table_'+str(extent)+'_'+str(year)+'_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.csv',index=False)
else:
    sum_table.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Figures_tables/'+str(date)+'/Summary_table_'+str(extent)+'_'+str(year)+'_lakemodel'+str(lakemodel)+'.csv',index=False)
del df1,cols

# Create an elevation banded summary table
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/All_watersheds/')
df1 = pd.read_csv('Survey_dates.csv')
df1['year'] = pd.to_numeric(df1['year'])
df1 = df1[df1['watershed']==str(watershed)]
df1 = df1[df1['year']==pd.to_numeric(year)]
df1 = df1.drop(columns='watershed')
df_elev = []
for a in range(len(subbasin)): 
    all_merged=[]
    for n in range(len(phases)):
        if glaciers == 'Y':
            file1=str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_Elevation_banded_mean_SWE.csv'
            file2=str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_Elevation_banded_total_SWV.csv'
        else:
            file1=str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_Elevation_banded_mean_SWE.csv'
            file2=str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/SWE_calculations/'+str(watershed)+'/Elevation_banded_water_volumes/'+str(year)+'/resolution_'+str(resolution2)+'m/lakemodel'+str(lakemodel)+'/'+str(subbasin[a])+'_'+str(year)+'_'+str(phases[n])+'_Elevation_banded_total_SWV.csv'
        result1=pd.DataFrame(pd.read_csv(file1)).set_index('elev_band')
        result2=pd.DataFrame(pd.read_csv(file2)).set_index('elev_band')
        last_col2 = result2.columns[-1]
        merged=result1.join(result2[last_col2], how='outer')
        merged=merged.drop('Unnamed: 0', axis=1)
        merged['SWE_mean_mm'] = merged['SWE_mean_mm'].round(0)
        merged['Total_SWV_m3'] = merged['Total_SWV_m3'].round(0)
        y = df1[df1['Survey']==str(phases[n])]
        merged['watershed'] = subbasin[a]
        merged['date'] = y.iloc[0]['date']
        all_merged.append(merged)
    x = pd.concat(all_merged)
    df_elev.append(x)
df_elev = pd.concat(df_elev)
if glaciers == 'Y':
    df_elev.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Figures_tables/'+str(date)+'/Elevation_table_'+str(extent)+'_'+str(year)+'_lakemodel'+str(lakemodel)+'_glaciermodel'+str(glaciermodel)+'.csv')
else:
    df_elev.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(year)+'/Figures_tables/'+str(date)+'/Elevation_table_'+str(extent)+'_'+str(year)+'_lakemodel'+str(lakemodel)+'.csv')

# Update the 'Yearly_comparison' csv
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/')
df1 = pd.read_csv(str(watershed)+'/All_years/Yearly_comparison.csv')
df2 = df1[df1["year"] != int(year)]
df2 = df2[['watershed','year','Survey','date','date_figure','Total_SWV_m3','Absolute_total_SWV_errors_m3','Mean_SWE_mm','Absolute_mean_SWE_error']]
df3 = pd.read_csv('All_watersheds/Survey_dates.csv')
df3 = df3[df3["watershed"] == str(watershed)]
df3 = df3[df3["year"] == int(year)]
df3 = df3.drop(['watershed','year'], axis = 1)
df4 = sum_table[['Survey','year','watershed','Total_SWV_m3','Absolute_total_SWV_errors_m3','Mean_SWE_mm','Absolute_mean_SWE_error']]
df5 = pd.merge(df3,df4,how='right',on=['Survey'])
df5 = df5[['watershed','year','Survey','date','date_figure','Total_SWV_m3','Absolute_total_SWV_errors_m3','Mean_SWE_mm','Absolute_mean_SWE_error']]

df_yearly = pd.concat([df2,df5])
df_yearly.to_csv(str(watershed)+'/All_years/Yearly_comparison.csv',index=False)
df1.to_csv(str(watershed)+'/All_years/Yearly_comparison_old.csv',index=False)

### PLOTS -------------------------------------------------------------------------
df_plot = df_elev.reset_index()
df_plot['watershed'].loc[df_plot['watershed']=="RC"] = 'Russell Creek'
df_plot['watershed'].loc[df_plot['watershed']=="CRU"] = 'Cruickshank'
df_plot['elev_band'].loc[df_plot['elev_band']==150] = '100-200'
df_plot['elev_band'].loc[df_plot['elev_band']==250] = '200-300'
df_plot['elev_band'].loc[df_plot['elev_band']==350] = '300-400'
df_plot['elev_band'].loc[df_plot['elev_band']==450] = '400-500'
df_plot['elev_band'].loc[df_plot['elev_band']==550] = '500-600'
df_plot['elev_band'].loc[df_plot['elev_band']==650] = '600-700'
df_plot['elev_band'].loc[df_plot['elev_band']==750] = '700-800'
df_plot['elev_band'].loc[df_plot['elev_band']==850] = '800-900'
df_plot['elev_band'].loc[df_plot['elev_band']==950] = '900-1000'
df_plot['elev_band'].loc[df_plot['elev_band']==1050] = '1000-1100'
df_plot['elev_band'].loc[df_plot['elev_band']==1150] = '1100-1200'
df_plot['elev_band'].loc[df_plot['elev_band']==1250] = '1200-1300'
df_plot['elev_band'].loc[df_plot['elev_band']==1350] = '1300-1400'
df_plot['elev_band'].loc[df_plot['elev_band']==1450] = '1400-1500'
df_plot['elev_band'].loc[df_plot['elev_band']==1550] = '1500-1600'
df_plot['elev_band'].loc[df_plot['elev_band']==1650] = '1600-1700'
df_plot['elev_band'].loc[df_plot['elev_band']==1750] = '1700-1800'
df_plot['elev_band'].loc[df_plot['elev_band']==1850] = '1800-1900'
df_plot['elev_band'].loc[df_plot['elev_band']==1950] = '1900-2000'
df_plot['elev_band'].loc[df_plot['elev_band']==2050] = '2000-2100'

ylist=list(df_plot['elev_band'].unique())
sns.set_style('whitegrid')
g = sns.catplot(data=df_plot,y='elev_band',x='SWE_mean_mm',col='watershed',col_wrap=2,hue='date',kind='bar',order=ylist[::-1])
g.set_axis_labels("SWE (mm)","Elevation (m)", fontsize = 12)
g.set_xticklabels(rotation=0)
g.set_titles(col_template="{col_name}",size=14)
sns.move_legend(
    g,"upper left", bbox_to_anchor=(.7,.1), title="", fontsize="large")
plt.savefig(
    f"{drive}:/LiDAR_data_processing/{lidar}/Final_products/{watershed}/{year}/Figures_tables/{date}/"
    f"Elevation_meanSWE.png",bbox_inches='tight', pad_inches=0.1)
