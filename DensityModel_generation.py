# This script develops a random forest snow density model, based on field snowdensity and snowdepth data; meteo, canopy, and topographic parameters
# To change the input variables, change the 'variables2' object in Line 168
# make sure that you save the model in the correct folder path and correct name (line 195)

watershed='MV' # Enter prefix for watershed of interest (ENG/CRU/TSI/MV)
subbasin='SEY' #Enter prefix for subbasin. If entire watershed is processed, repeat watershed prefix
year=['2021','2022','2023','2024'] # Enter the years of field data to be included in density model
drive = 'K'
lidar = 'ACO' # Enter 'ACO' for a survey by plane or 'RPAS' for a survey by drone

# Import libraries
import rasterio
import pandas as pd
import numpy as np
import math
import glob
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Import field data for all field surveys for each years 
all_years=[]
for k in range(len(year)):
    filenames=sorted(glob.glob(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Plot_characteristic_sampling/'+str(year[k])+'/All_plot_characteristics_'+str(subbasin)+'*.csv')) # Path to input csv files
    variables=["depth","slope","canopy_cover","canopy_height","aspect","elevation","Xt","curvature","PDD_sum","Total_snowfall","density"]
    all_surveys=[]
    for n in range(len(filenames)):
        all_variables=[]
        for m in range(len(variables)):
            x=np.ndarray.flatten(np.array(pd.read_csv(filenames[n],usecols=[variables[m]])))
            all_variables.append(x)
        all_surveys.append(all_variables)
    all_years.append(all_surveys)
del filenames,variables,m,x,k,all_variables,all_surveys

# Remove data points where density is 3 SDs outside the mean or below 10%
for k in range(len(all_years)):
    all_surveys=all_years[k]
    for n in range(len(all_surveys)):
        x=all_surveys[n]
        mean=np.mean(x[10])
        sd=np.std(x[10])
        upper_limit=mean+(3*sd)
        lower_limit=mean-(3*sd)
        upper_outliers=np.asarray(np.where(x[10]>upper_limit))
        lower_outliers=np.asarray(np.where(x[10]<lower_limit))
        sub10_outliers=np.asarray(np.where(x[10]<0.1))
        all_variables=[]
        for m in range(len(x)):
            xx=x[m]
            xx=np.delete(xx,upper_outliers)
            xx=np.delete(xx,lower_outliers)
            xx=np.delete(xx,sub10_outliers)
            all_variables.append(xx)
        all_surveys[n]=all_variables
    all_years[k]=all_surveys
del all_surveys,n,m,k,x,xx,mean,sd,upper_limit,lower_limit,upper_outliers,lower_outliers,sub10_outliers,all_variables

# Merge all surveys together for each year
for k in range(len(all_years)):
    all_surveys=list(np.transpose(np.array(all_years[k],dtype='object')))
    all_variables=[]
    for n in range(len(all_surveys)):
        x=list(all_surveys[n])
        z=[]
        for m in range(len(x)):
            y=np.reshape(x[m],(len(x[m]),1))
            z.append(y)
        zz=np.vstack(z)
        all_variables.append(zz)
    all_years[k]=all_variables
del all_surveys,all_variables,y,z,zz,n,m,k,x

# Merge all years together
zz=[]
for n in range(len(all_years[0])):
    z=[]
    for m in range(len(all_years)):
        x=all_years[m]
        y=x[n]
        z.append(y)
    zz.append(z)
all_variables=[]
for n in range(len(zz)):
    x=zz[n]
    y=np.vstack(x[:])
    all_variables.append(y)
del n,x,y,z,zz,all_years

# Transform semi-sinusoidal data
aspect=all_variables[4]
aspect_x=[]
aspect_y=[]
for i in range(len(aspect)):
    x=math.sin(2*math.pi*aspect[i]/360)
    y=math.cos(2*math.pi*aspect[i]/360)
    aspect_x.append(x)
    aspect_y.append(y)
del aspect,i,x,y
aspect_x=np.array(aspect_x).reshape(len(aspect_x),1)
aspect_y=np.array(aspect_y).reshape(len(aspect_y),1)

# Reformat data
variables=[all_variables[0],all_variables[1],all_variables[2],all_variables[3],aspect_x,aspect_y,all_variables[5],all_variables[6],all_variables[7],all_variables[8],all_variables[9],all_variables[10]]
final_variable_names=['Snow depth','Slope','Canopy cover','Canopy height','Eastfacingness','Northfacingness','Elevation','Xt (meteorological metric)','Curvature','PDD sum','Snowfall','Snow density']
del all_variables,aspect_x,aspect_y

#Set NaN values in canopy data to zero
canopy_indices=[2,3]
for n in range(len(canopy_indices)):
    x=canopy_indices[n]
    nans=np.argwhere(np.isnan(variables[x]))
    y=variables[x]
    y[nans]=0
    variables[x]=y
x=variables[3]
nans=np.where(x>100)
x[nans]=0
variables[3]=x
del canopy_indices,n,nans,x,y

# Remove rows with unrealistic snow depths
nans=np.where(variables[0]>1000)
for n in range(len(variables)):
    variables[n]=np.delete(variables[n],nans)
del nans,n

# Remove rows with unrealistic snow densities
nans=np.where(variables[11]>0.8)
for n in range(len(variables)):
    variables[n]=np.delete(variables[n],nans)
nans=np.where(variables[11]<0)
for n in range(len(variables)):
    variables[n]=np.delete(variables[n],nans)
del nans,n

# Remove remaining rows where nans are present in any of the variables
for n in range(len(variables)):
    x=variables[n]
    nans=np.argwhere(np.isnan(x))
    for m in range(len(variables)):
        y=variables[m]
        z=np.delete(y,nans)
        variables[m]=z
del nans,n,m,x,y,z
        
# Remove rows with infinite values
infs=np.argwhere(np.isinf(variables[11]))
for n in range(len(variables)):
    x=variables[n]
    y=np.delete(x,infs)
    variables[n]=y
del n,x,y,infs

# Prepare testing and training data for RF model
v0=variables[0] #Snow depth
v1=variables[1] #Slope
v2=variables[2] #Canopy cover
v3=variables[3] #Canopy height
v4=variables[4] #Eastfacingness
v5=variables[5] #Northfacingness
v6=variables[6] #Elevation
v7=variables[7] #Xt (meteorological metric)
v8=variables[8] #Curvature
v9=variables[9] #PDD sum
v10=variables[10] #Snowfall
variables2=[v0,v1,v4,v5,v6,v7,v8]

# Normalise input data
from sklearn.preprocessing import MinMaxScaler
all_scalers=[]
for n in range(len(variables2)):
    x=variables2[n]
    x=x.reshape(len(x),1)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(x)
    X_normalized=X_normalized.reshape(len(X_normalized),)
    variables2[n]=X_normalized
    all_scalers.append(scaler)
    
# Reformat data
X=np.transpose(variables2)
Y=variables[11] #Snow density
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# Create RF model
model_rf = RandomForestRegressor(n_estimators=500,max_depth=5,min_samples_split=20,min_samples_leaf=10,max_features=0.3,n_jobs=-1,random_state=12345)
model_rf.fit(X_train, Y_train)
pred_test_rf = model_rf.predict(X_test)
rmse_rf=np.sqrt(mean_squared_error(Y_test,pred_test_rf))
syst_error=np.mean(pred_test_rf-Y_test)
rand_error=np.std(pred_test_rf-Y_test)

# Output ---------------------------------------------------------------------------------------------------------------
# Save RF model
os.chdir(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Overall_density_model/No_Canopy')
joblib.dump(model_rf, "RF_density_model_"+str(watershed)+"_NoCan_2021_2024.joblib")

# Export normalization scalers
for n in range(len(all_scalers)):
    scaler=all_scalers[n]
    joblib.dump(scaler, 'scaler'+str(n+1)+'.pkl')

# Export error values
all_errors= pd.DataFrame({'rmse':[rmse_rf],'syst_error':[syst_error],'rand_error':[rand_error]})
all_errors.to_csv(str(drive)+':/LiDAR_data_processing/'+str(lidar)+'/Density_modelling/'+str(watershed)+'/Overall_density_model/No_Canopy/model_error_values.csv')