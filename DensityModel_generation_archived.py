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