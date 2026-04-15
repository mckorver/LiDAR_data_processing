
# Import packages
import rasterio
import rasterio.features
from shapely.geometry import Point
import os
import numpy as np
import pandas as pd

# Import processing variables
var = pd.read_csv('K:/LiDAR_data_processing/Field_data/Field_data_processing_variables.csv', 
                  dtype={'years':str, 'resolution1':str, 'resolution2':str, 'BEversion':str, 'CANversion':str})
watershed = var['watershed'][0]
drive = var['drive'][0]
lidar = var['lidar'][0]
resolution1 = var['resolution1'][0]
resolution2 = var['resolution2'][0]
BEversion = var['BEversion'][0]
CANversion = var['CANversion'][0]
def append_fun(a,b):
    x = var[b][var[b].notna()]
    for n in range(len(x)):
        y = x[n]
        if isinstance(y, float):
            a.append(int(y))
        else:
            a.append(y)
years_or=[]
phase=[]
append_fun(years_or,'years')
append_fun(phase,'phases')
phases_or=[]
for n in range(len(phase)):
    a=[]
    for m in range(phase[n]):
        x='P'+str(m+1)
        a.append(x)
    phases_or.append(a)

years= years_or[1:7]
phases= phases_or[1:7]

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
        #core=np.array(pd.read_csv('Field_data_'+str(watershed)+'_'+str(years[a])+'_'+str(phases[a][b])+'.csv', usecols=['core_length_final'])).astype('float64')
        #core=core.reshape(len(core),)
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
        #cores.append(core)
        densities.append(density)
        manual_remove.append(manual_rem)
        field_phases.append(field_phase)

df=pd.DataFrame({"phase":np.concatenate(field_phases),"datetime_field":np.concatenate(datetimes_field),"datetime_aco":np.concatenate(datetimes_aco),
                 "easting_m":np.concatenate(eastings),"northing_m":np.concatenate(northings),
                 "plot_id":np.concatenate(plot_ids),"plot_type":np.concatenate(plot_types),"cardinal":np.concatenate(card_dirs),
                 "snow_depth":np.concatenate(depths),#"core_depth":np.concatenate(cores),
                 "density":np.concatenate(densities), "manual_remove":np.concatenate(manual_remove)})
df['datetime_field'] = df['datetime_field'].astype('datetime64[ns]')
df['datetime_aco'] = df['datetime_aco'].astype('datetime64[ns]')
#df['time_gap_hr']=df['datetime_field']-df['datetime_aco']
#df['time_gap_hr'] = df['time_gap_hr'].dt.total_seconds()/3600
#df['time_gap_hr'] = df['time_gap_hr'].abs()
df['year'] = df['datetime_field'].dt.year.astype('string')
df['snow_depth_m'] = df['snow_depth']/100

# QAQC field data
df=df[(df['density'].notna())&(df['density']>0)]
df['flag'] = 'AV'
#df.loc[(df['time_gap_hr']>60), 'flag'] = 'time'
df.loc[(df['manual_remove']=='Y'), 'flag'] = 'manual'
df.loc[(df['density']>0.8)|(df['density']<0.1), 'flag'] = 'range'
#df.loc[(df['snow_depth']-df['core_depth']<-5)|(df['snow_depth']/df['core_depth']>=2), 'flag'] = 'core'
filt=df.loc[(df['flag']=='AV')].copy()

# Separate between cardinal field plots and point field plots
filt['swe']=filt.density*filt.snow_depth*10
temp=filt.drop(columns=['flag','manual_remove'])
grouped=temp.loc[(temp['plot_type']=='Cardinal 10 m')]
grouped=grouped[['datetime_field','datetime_aco','easting_m','northing_m','phase','cardinal','plot_id','plot_type','year','density','snow_depth_m','swe']]
grouped['easting_m']=np.where(grouped['cardinal']!='Centre',np.nan,grouped['easting_m'])
grouped['northing_m']=np.where(grouped['cardinal']!='Centre',np.nan,grouped['northing_m'])
times = grouped.loc[(grouped['cardinal']=='Centre')][['datetime_field','datetime_aco','plot_type','year','phase','plot_id']]
grouped=grouped.groupby(['phase','plot_id','year']).mean().reset_index()
grouped=pd.merge(grouped,times,on=['phase','plot_id','year'],how='right')
grouped=grouped[['datetime_field','datetime_aco','year','phase','plot_id','plot_type','easting_m','northing_m','density','snow_depth_m','swe']]
#points=temp.loc[(temp['plot_type']=='Point')]
#points=points[['datetime_field','datetime_aco','year','phase','plot_id','plot_type','easting_m','northing_m','density','snow_depth_m','swe']]
# Combine points and grouped data
#grouped_final=pd.concat([grouped,points])
del temp

# Save field data
os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/all_years/')
grouped.to_csv('Field_data_'+str(watershed)+'_plot_averaged.csv',index=False)
del datetimes_field,eastings,northings,depths,densities,manual_remove,field_phases

x = grouped[grouped['year']=='2026']
years= years_or[0:6]
phases= phases_or[0:6]
plot_ids= x['plot_id'].tolist()
eastings= x['easting_m'].tolist()
northings= x['northing_m'].tolist()
## Import LiDAR derived input variables and sample for GROUPED field locations
#file=Path(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/all_years/LiDAR_data_'+str(watershed)+'_field_plots.csv')
#if file.is_file():
#    os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/all_years/')
#    pd_lidar=pd.read_csv('LiDAR_data_'+str(watershed)+'_field_plots.csv')
#    pd_lidar['year'] = pd_lidar['year'].astype(str)
#else:
buffer_distance = 10 # buffer distance around centre point of cardinal plot
# Import snow depth parameters
lidar_snow=[]
for m in range(len(years)):
    list3=[]
    os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Final_products/'+str(watershed)+'/'+str(years[m])+'/Maps/')
    for n in range(len(phases[m])):
        list2=[]
        src_snow = rasterio.open('SnowDepth/resolution_'+str(resolution1)+'m/Final/'+str(watershed)+'_'+str(years[m])+'_'+str(phases[m][n])+'_SnowDepth_lakemodelN.tif')
        src_swe = rasterio.open('SWE/resolution_'+str(resolution2)+'m/Final/'+str(watershed)+'_'+str(years[m])+'_'+str(phases[m][n])+'_SWE.tif') 
        for b in range(len(plot_ids)):
            list1 = []
            list1.append(years[m])
            list1.append(phases[m][n])
            list1.append(plot_ids[b])
            coords = (eastings[b], northings[b])
            point = Point(coords)
            buffer_poly = point.buffer(buffer_distance)
            # Create a mask from the buffer geometry. geometry_mask returns True for outside, False for inside
            mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src_snow.shape,transform=src_snow.transform,invert=False)
            data_snow = src_snow.read(1) # Read data and apply mask. Read first band
            sampled_values = data_snow[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
            sampled_value = sampled_values.mean() # Take the mean of all pixels within buffer
            list1.append(sampled_value)
            mask = rasterio.features.geometry_mask([buffer_poly],out_shape=src_swe.shape,transform=src_swe.transform,invert=False)
            data_swe = src_swe.read(1) # Read data and apply mask. Read first band
            sampled_values = data_swe[~mask] # Extract only the masked pixels (where mask is False, i.e., inside buffer). Using ~mask to select True where the buffer is
            sampled_value = sampled_values.mean() # Take the mean of all pixels within buffer
            list1.append(sampled_value)
            list2.append(list1)
            print("plot "+ plot_ids[b] + "done, year "+ years[m]+", phase "+phases[m][n])
        list3.extend(list2)
    lidar_snow.extend(list3)
pd_lidar=pd.DataFrame(lidar_snow,columns=['year','phase','plot_id','LiDAR_snow_depth', 'LiDAR_SWE'])

os.chdir(str(drive)+':/LiDAR_data_processing/Field_data/'+str(watershed)+'/all_years/')
pd_lidar.to_csv('LiDAR_data_'+str(watershed)+'_field_plots.csv',index=False)