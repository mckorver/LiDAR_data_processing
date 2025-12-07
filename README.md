<<<<<<< HEAD
LiDAR-based snowdepth processing, snowdensity modelling, and SWE calculations

See "LiDAR_data_processing\General_information\LiDAR_data_processing_SOPs_v2.docx" for a word version (with figures) of this file.

Introduction
This is a step-by-step guide to calculate snow depth, snow density, and snow water equivalent from LiDAR surveys, using the python scripts stored on Github:  (https://github.com/mckorver/LiDAR_data_processing).
These scripts run smoothly when using Python 3.7.0 and anaconda (https://www.anaconda.com/download). QGIS is used for visual inspections and few calculations.
Set up python environment and install packages in this order (use anaconda command line):
1.	conda create -n snow-env python=3.7.0
2.	conda activate snow-env
3.	conda install gdal==3.0.2
4.	pip install opencv-contrib-python
5.	pip install pyrsgis==0.4.1
6.	conda install matplotlib==3.5.2
7.	conda install pandas==1.3.5
8.	conda install joblib==1.1.1
9.	conda install scikit-learn==0.24.2
The entire workflow consists of six steps in 13 scripts: preparation, snowdepth processing, snowdensity model generation, Snowdensity model inference, SWE calculations, and field validation & deliverables.
All scripts start with a section where you need to indicate:
-	Watershed
-	Subbasin(s)
-	Survey Year
-	Survey Phase(s).
-	The bare earth version (currently, only one version is used)
-	The raster resolution in m
-	The computer drive used
-	The type of LiDAR survey (ACO or RPAS)

Usually, multiple phases and subbasins can be run at the same time. If not, it will be indicated in the scripts.
 
1 Preparation

1.1 Download LiDAR and ortho data
Hakai stores the LiDAR and ortho data on an FTP server:
Host: 69.196.72.195
Port: 22
User: viuuser
Password: cryosphere

1.2 Register survey date
Surveys are indicated by a ‘phase number’, i.e., each year during the melt season an x number of surveys are completed, indicated by P1, P2, P3 etc. Register which date belongs to which phase number in “LiDAR_data_processing/General_information/LiDAR_survey_dates.xlsx”.

1.3 Resample LiDAR data to the same extents (+resolution)
Scripts only work if all raster data have the same extent and resolution. For the ACO surveys, we use the 1 m resolution snow-free bare earth raster to set the extent of all other input data. Snow-free bare earth data can be found in “LiDAR_data_processing/Bare_earth/[Watershed]/[version]/[resolution]/[Watershed]_BE_[version]_[resolution].tif”. Load LiDAR and snow-free bare earth data into QGIS. In Raster Calculator, load the LiDAR data into the Raster Calculator Expression box. Then select the snow-free bare earth data and click Use Selected Layer Extent. Make sure that the output CRS matches the project CRS. Save new LiDAR data as “LiDAR_data_processing/ACO/Input_data/LiDAR_data/[Watershed]/[Year]/[Watershed]_[Year]_[Phase]_SS.tif.

1.4 Find snow free elevation thresholds
Load ortho map in GIS along with snow-free bare earth to find the snow-free elevation threshold – i.e. the highest elevation below which there is minimal snow present (there could be some snow free areas above this elevation threshold and a few small snow patches below). Add results to a csv with column headers “Survey” and “SFETs” and save csv to “LiDAR_data_processing/ACO/Snow_depth_processing/[Watershed]/Snow_free_elevation_masks” with file name “SFETs_[Watershed]_[year].csv”. Note that in the past, snow-free elevation areas were also saved as TIFs. Scripts are currently not set up to process data using these maps, only fixed elevation thresholds.

1.5 Download and format weather station data
The machine learning model uses the meteorological parameter ‘Xt’ as an input parameter. This is calculated from air temperature and total precipitation data (Sep 1st – day of survey) of a high and a low elevation weather station within the watershed (see Bisset et al., 2025). Use the following weather station data for the ACO surveys:
Englishman watershed:
•	High elevation: Mount Arrowsmith downloaded from the CHRL website (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_mountarrowsmith/Export)
•	Low elevation: Qualicum Beach Airport (https://climate.weather.gc.ca/climate_data/hourly_data_e.html?StationID=45627)
Cruickshank watershed:
•	High elevation: Upper Cruickshank (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_uppercruikshank/Export).
•	Low elevation Perseverance (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_perseverance/Export).
Tsitika:
•	High elevation: Steph 2 (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_steph2/Export).
•	Low elevation: Steph 6 (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_steph6/Export).
Metro Vancouver:
•	High elevation: Orchid Lake (https://aqrt.nrs.gov.bc.ca/Data/DataSet/Summary/Location/3A19P/DataSet/TA/Telemetry/Interval/Latest)
•	Low elevation: Palisade Lake (https://aqrt.nrs.gov.bc.ca/Data/DataSet/Summary/Location/3A09P/DataSet/TA/Telemetry/Interval/Latest)
Raw weather station data are saved in “LiDAR_data_processing/ACO/Input_data/Weather_station_data/[Watershed]”. Then, two formatted csv files are prepared and saved in “LiDAR_data_processing\ACO\Density_modelling\[Watershed]\Meteorological_parameter_modelling\[Year]”. 
1)	The meteo data in a csv file named “WS_data_[Watershed]_[Year].csv”. Formatting needs to match the example below. The lower weather station total precipitation (PC_low) data is currently not used in the calculations but stored for potential future updates of the calculations (i.e., using a precipitation lapse rate). Generally, we start the timeseries on Sep 1st of the preceding year (when snow might start falling at high elevations). However, Oct 1st has also been used.
2)	A metadata file named “Metadata_[Watershed]_[Year]”.csv, See example below. The ‘survey_days’ are the number of days between the first day on record (step 1; Sep 1st or Oct 1st) and the day of the survey. In this example below, Sep data was not available so the number of days were counted from Oct 1st. The “calculate_survey_dates.csv” (Density_modelling\[Watershed]\Meteorological_parameter_modelling\) is a tool to help you calculate the no. of days.

1.6 The 0_prep.py script
The 0_prep.py script is not meant to be run in its entirety and is not mandatory. It contains code chunks that can 1) downsample rasters (final data products are generally delivered at 2m resolution, downsampled from 1m, to reduce computing power) and 2) set the extent and clip rasters to a study area boundary. The extent and clipping actions have been applied to the drone data specifically and are not relevant to ACO processing.

2 Snowdepth processing
Four snowdepth processing scripts output several ‘provisional’ snowdepth maps. Not all processing scripts are mandatory. Depending on the quality of the data, you can decide at the end which processed map you select as your final snowdepth map. Copy this map and save it as “LiDAR_data_processing\ACO\Final_products\Maps\SnowDepth\[Watershed]\[Year]\[resolution]\[Watershed]_[Year]_[Phase]_SnowDepth.tif”.

2.1 SnowDepth_initial_processing.py
This code loads LiDAR data, calculates snow depths, and checks the bias on snow-free roads. It outputs unprocessed snowdepth maps in “LiDAR_data_processing\ACO\Snow_depth_processing\[Watershed]\Provisional”and bias analysis results in “K:\LiDAR_data_processing\ACO\Bias_analysis\[Watershed]\[Year]”. If the bias is high (i.e., >10 cm) it is possible that the LiDAR data needs to be dropped or lifted to control points (for the ACO surveys, this is Hakai’s responsibility).
Inspect the snowdepth map in QGIS. If you see areas with very noisy data that need to be clipped out (usually along the periphery), manually draw a polygon over that area and rasterize it with pixel values set to 1, and NoData values set to 0. Make sure that this raster has the same resolution and extent as the LiDAR & bare earth rasters. Save this raster as “LiDAR_data_processing\ACO\Snow_depth_processing\[Watershed]\Peripheral_masks\[Watershed]_[Year]_[Phase]_PeriMask.tif”. Note that if the noisy data falls outside the watershed boundary, masking is not necessary. If you see no noisy data, no action is required. If you see a lot of noisy throughout the watershed, it is possible that the LiDAR data needs to be reassessed and reprocessed by Hakai.

2.2 SnowDepth_corrections.py
This code clips out noisy areas (PeriMask), lakes, unrealistic negative values (<-5m), and noise introduced by vegetation misclassification or sensor issues. It also caps values >-5 & <0 to 0m and unrealistic high values to 10m + mean snowdepth. Noise introduced by vegetation misclassification or LiDAR sensor issues appear as ‘bumps’ or ‘throughs’ on the map. We clip out these bumps and throughs by applying a smoothing filter. You have to set the variables for this filter, which decide how aggressively it is applied:
-	Avalanche threshold. You want to avoid the removal of avalanches, which are ‘bumps’. Visually inspect the ortho data in QGIS to decide above which elevation avalanches appear. The filter is not applied above this elevation.
-	Upper and lower detection thresholds: the change in snowdepth that is assumed to be anomalous between neighboring areas
-	Kernel size: the size of the kernel smoothing moving average window
-	Expansion distance: the amount of pixels to be removed around an anomaly.  
The script outputs two processed snowdepth maps, one with and one without vegetation noise corrections. After visual inspection of both maps in QGIS, decide if, and with what settings you want to apply the vegetation noise corrections. Final smoothing variables are saved into “LiDAR_data_processing\ACO\Snow_depth_processing\[Watershed]\Processing_variables\”.

2.3 SnowDepth_gapfilling_interpolation.py & SnowDepth_gapfilling_modelling.py
These scripts fill gaps that were originally present in the LiDAR data or that were created by the noise corrections, but do not fill lake areas. The Cruickshank glaciers, however, are filled. Generally, small gaps are filled with interpolation, whereas larger gaps (such as glaciers, or large areas of vegetation misclassifications) are filled using modelling.

3 Snowdensity model generation
It is not mandatory to generate a new density model every time. If you do not need to update the density model, skip scripts 5 and 6 and go to the next section.

4 Snowdensity model inference
Snow density is estimated with a machine learning random forest model based on in-situ snowdensity & snowdepth, elevation (snow-free bare earth), slope, northness, eastness, curvature, canopy cover, canopy height, and Xt. The slope, northness, eastness, and curvature maps are calculated from elevation (snow-free bare earth), which can be done in QGIS: https://medium.com/@valentino.pintar/how-to-prepare-predictors-for-ecological-modelling-e442601120e3. Canopy cover and canopy height maps are prepared by Hakai. Xt will be prepared in the following step. 

4.1 SnowDensity_meteo_parameter_modelling
This script calculates Xt, a parameter used for snowdensity modelling. Xt is calculated from air temperature (positive degree days) and snowfall measured at one high and one low elevation meteorological station. The total precipitation data might or might not have to be QC’d yet, depending on the source of the meteo data. Uncomment the corrections that you want to apply (starting line 49). Note that the script assumes precip in mm and air temp in Celcius. You need to set an assumed rain/snow temperature threshold, which according to Jennings et al., 2018, is 0.97 for Metro Vancouver, 0.98 for Tsitika, 0.90 for Cruikshank, and 0.91 for Englishman watersheds.
This script outputs raster maps of Xt, Snowfall, and positive degree days and csv files of Xt, Snowfall, and positive degree days by elevation in “LiDAR_data_processing\ACO\Density_modelling\[Watershed]\Meteorological_parameter_modelling\[Year]\Output”. Currently, only Xt is used in the snow density model.

4.2 SnowDensity_model_inference
This script applies the machine learning random forest model to calculate snow density. The output is a snow density raster map, saved to “LiDAR_data_processing\ACO\Final_products\Maps\SnowDensity\[Watershed]\[Year]\[resolution]\[Watershed]_[Year]_[Phase]_SnowDensity.tif”.
If you do not have canopy data for your watershed (i.e., as for Metro Vancouver), use the “_NoCanpy” script. Otherwise, use the “_AllParameters” script. In the first code block (line 24-31), check that the right density model is being loaded. 
Make sure that all input rasters have the same extent, resolution, and project CRS. Because of computational limitations, we generally downscale all input maps to a 2m resolution and generate snow density maps at 2m resolution as well. The 0_prep.py script can perform the downscaling. 

5 SWE calculations
Snow water equivalent (SWE) is the product of snow depth and snow density. The following scripts calculate SWE, summary values, and uncertainties.

5.1 SWE_calculations
This script calculates and outputs a SWE raster map, and calculates subbasin summary values, including the mean snow depth (m), the mean snow depth above the snowline (m), the mean snow density (kg/m3), the mean SWE depth (mm), the total snow volume (m3), and the total snow water volume (m3). The SWE map is saved to “LiDAR_data_processing\ACO\Final_products\Maps\SWE\[Watershed]\[Year]\[resolution]\[Watershed]_[Year]_[Phase]_SWE.tif”, and the summary values are saved to “LiDAR_data_processing/ACO/SWE_calculations/[Watershed]/Key_numbers/[year]/[resolution]”.

5.2 SWE_elevation_banded_calculations
This script takes the SWE raster map(s) calculated in step 5.1 and calculates the mean SWE and total snow water volume by 100 m elevation bands. Results are added to “LiDAR_data_processing/ACO/SWE_calculations/[Watershed]/Elevation_banded_water_volumes/[year]/[resolution]/”.

5.3 SWE_uncertainty_propagation
This script calculates the percent and absolute uncertainty in total water volume (m3). The percent uncertainty can be applied to mean SWE to calculate the uncertainty in SWE. It is based, in part, on a random density model error (RDME), which is a value associated to the random forest model used for density calculations. The last recorded model runs by Rosie provided RDMEs of Cruikshank = 72, Englishman = 56, Metro Vancouver = 34, Tsitika = 51 kg/m3. For future updates of these models, the associated RDME will be saved in a file called ‘model_error_values.csv’ within the folder of that specific model (“LiDAR_data_processing\ACO\Density_modelling\[Watershed]\Overall_density_model”). Moreover, uncertainty is also based on errors in snow depth in areas below the snowline (i.e., pixel values deviating from 0 m). If the entire watershed is covered by snow, uncertainty calculations can not be performed and will be output as ‘0’ uncertainty.
Uncertainty is calculated based on the entire watershed. The same percent uncertainty is then applied to each subbasin. Uncertainty values are exported to “LiDAR_data_processing/ACO/SWE_calculations/[Watershed]/Key_numbers/[year]/[resolution]”.

6 Field checks and deliverables

6.1 Field_validation
This script compares snow depth, snow density, and SWE outputs to field observations that were made on the same day as (or only a few days removed from) the LiDAR survey. Multiple field measurements are taken at certain plots, usually measurements spaced about a meter apart in different cardinal directions. Generally, we assess plot averages, not single measurements. This script outputs csv files with plot-averaged statistics (i.e., mean and sd of field measurements in plot & mean and sd of LiDAR derived pixel values in plot) and ‘difference’ statistics (i.e., assessing the mean, sd, and rmse of differences (LiDAR – field).

6.2 Deliverables
This script gathers the various outputs generated throughout the workflow and merges them into summary tables. Under ‘date’, enter the date of today. All summary tables are output in “LiDAR_data_processing\ACO\Final_products\Tables\[Watershed]\[date]”. R scripts are available to generate summary figures from these summary tables, also available on the Github repository as “Figures_[Watershed].R”.
=======
LiDAR-based snowdepth processing, snowdensity modelling, and SWE calculations

See "LiDAR_data_processing\General_information\LiDAR_data_processing_SOPs_v2.docx" for a word version (with figures) of this file.

Introduction
This is a step-by-step guide to calculate snow depth, snow density, and snow water equivalent from LiDAR surveys, using the python scripts stored on Github:  (https://github.com/mckorver/LiDAR_data_processing).
These scripts run smoothly when using Python 3.7.0 and anaconda (https://www.anaconda.com/download). QGIS is used for visual inspections and few calculations.
Set up python environment and install packages in this order (use anaconda command line):
1.	conda create -n snow-env python=3.7.0
2.	conda activate snow-env
3.	conda install gdal==3.0.2
4.	pip install opencv-contrib-python
5.	pip install pyrsgis==0.4.1
6.	conda install matplotlib==3.5.2
7.	conda install pandas==1.3.5
8.	conda install joblib==1.1.1
9.	conda install scikit-learn==0.24.2
The entire workflow consists of six steps in 13 scripts: preparation, snowdepth processing, snowdensity model generation, Snowdensity model inference, SWE calculations, and field validation & deliverables.
All scripts start with a section where you need to indicate:
-	Watershed
-	Subbasin(s)
-	Survey Year
-	Survey Phase(s).
-	The bare earth version (currently, only one version is used)
-	The raster resolution in m
-	The computer drive used
-	The type of LiDAR survey (ACO or RPAS)

Usually, multiple phases and subbasins can be run at the same time. If not, it will be indicated in the scripts.
 
1 Preparation

1.1 Download LiDAR and ortho data
Hakai stores the LiDAR and ortho data on an FTP server:
Host: 69.196.72.195
Port: 22
User: viuuser
Password: cryosphere

1.2 Register survey date
Surveys are indicated by a ‘phase number’, i.e., each year during the melt season an x number of surveys are completed, indicated by P1, P2, P3 etc. Register which date belongs to which phase number in “LiDAR_data_processing/General_information/LiDAR_survey_dates.xlsx”.

1.3 Resample LiDAR data to the same extents (+resolution)
Scripts only work if all raster data have the same extent and resolution. For the ACO surveys, we use the 1 m resolution snow-free bare earth raster to set the extent of all other input data. Snow-free bare earth data can be found in “LiDAR_data_processing/Bare_earth/[Watershed]/[version]/[resolution]/[Watershed]_BE_[version]_[resolution].tif”. Load LiDAR and snow-free bare earth data into QGIS. In Raster Calculator, load the LiDAR data into the Raster Calculator Expression box. Then select the snow-free bare earth data and click Use Selected Layer Extent. Make sure that the output CRS matches the project CRS. Save new LiDAR data as “LiDAR_data_processing/ACO/Input_data/LiDAR_data/[Watershed]/[Year]/[Watershed]_[Year]_[Phase]_SS.tif.

1.4 Find snow free elevation thresholds
Load ortho map in GIS along with snow-free bare earth to find the snow-free elevation threshold – i.e. the highest elevation below which there is minimal snow present (there could be some snow free areas above this elevation threshold and a few small snow patches below). Add results to a csv with column headers “Survey” and “SFETs” and save csv to “LiDAR_data_processing/ACO/Snow_depth_processing/[Watershed]/Snow_free_elevation_masks” with file name “SFETs_[Watershed]_[year].csv”. Note that in the past, snow-free elevation areas were also saved as TIFs. Scripts are currently not set up to process data using these maps, only fixed elevation thresholds.

1.5 Download and format weather station data
The machine learning model uses the meteorological parameter ‘Xt’ as an input parameter. This is calculated from air temperature and total precipitation data (Sep 1st – day of survey) of a high and a low elevation weather station within the watershed (see Bisset et al., 2025). Use the following weather station data for the ACO surveys:
Englishman watershed:
•	High elevation: Mount Arrowsmith downloaded from the CHRL website (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_mountarrowsmith/Export)
•	Low elevation: Qualicum Beach Airport (https://climate.weather.gc.ca/climate_data/hourly_data_e.html?StationID=45627)
Cruickshank watershed:
•	High elevation: Upper Cruickshank (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_uppercruikshank/Export).
•	Low elevation Perseverance (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_perseverance/Export).
Tsitika:
•	High elevation: Steph 2 (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_steph2/Export).
•	Low elevation: Steph 6 (https://galiano.islandhosting.com:2083/ go to: phpMyAdmin/viuhydro_wx_data_v2/qaqc_steph6/Export).
Metro Vancouver:
•	High elevation: Orchid Lake (https://aqrt.nrs.gov.bc.ca/Data/DataSet/Summary/Location/3A19P/DataSet/TA/Telemetry/Interval/Latest)
•	Low elevation: Palisade Lake (https://aqrt.nrs.gov.bc.ca/Data/DataSet/Summary/Location/3A09P/DataSet/TA/Telemetry/Interval/Latest)
Raw weather station data are saved in “LiDAR_data_processing/ACO/Input_data/Weather_station_data/[Watershed]”. Then, two formatted csv files are prepared and saved in “LiDAR_data_processing\ACO\Density_modelling\[Watershed]\Meteorological_parameter_modelling\[Year]”. 
1)	The meteo data in a csv file named “WS_data_[Watershed]_[Year].csv”. Formatting needs to match the example below. The lower weather station total precipitation (PC_low) data is currently not used in the calculations but stored for potential future updates of the calculations (i.e., using a precipitation lapse rate). Generally, we start the timeseries on Sep 1st of the preceding year (when snow might start falling at high elevations). However, Oct 1st has also been used.
2)	A metadata file named “Metadata_[Watershed]_[Year]”.csv, See example below. The ‘survey_days’ are the number of days between the first day on record (step 1; Sep 1st or Oct 1st) and the day of the survey. In this example below, Sep data was not available so the number of days were counted from Oct 1st. The “calculate_survey_dates.csv” (Density_modelling\[Watershed]\Meteorological_parameter_modelling\) is a tool to help you calculate the no. of days.

1.6 The 0_prep.py script
The 0_prep.py script is not meant to be run in its entirety and is not mandatory. It contains code chunks that can 1) downsample rasters (final data products are generally delivered at 2m resolution, downsampled from 1m, to reduce computing power) and 2) set the extent and clip rasters to a study area boundary. The extent and clipping actions have been applied to the drone data specifically and are not relevant to ACO processing.

2 Snowdepth processing
Four snowdepth processing scripts output several ‘provisional’ snowdepth maps. Not all processing scripts are mandatory. Depending on the quality of the data, you can decide at the end which processed map you select as your final snowdepth map. Copy this map and save it as “LiDAR_data_processing\ACO\Final_products\Maps\SnowDepth\[Watershed]\[Year]\[resolution]\[Watershed]_[Year]_[Phase]_SnowDepth.tif”.

2.1 SnowDepth_initial_processing.py
This code loads LiDAR data, calculates snow depths, and checks the bias on snow-free roads. It outputs unprocessed snowdepth maps in “LiDAR_data_processing\ACO\Snow_depth_processing\[Watershed]\Provisional”and bias analysis results in “K:\LiDAR_data_processing\ACO\Bias_analysis\[Watershed]\[Year]”. If the bias is high (i.e., >10 cm) it is possible that the LiDAR data needs to be dropped or lifted to control points (for the ACO surveys, this is Hakai’s responsibility).
Inspect the snowdepth map in QGIS. If you see areas with very noisy data that need to be clipped out (usually along the periphery), manually draw a polygon over that area and rasterize it with pixel values set to 1, and NoData values set to 0. Make sure that this raster has the same resolution and extent as the LiDAR & bare earth rasters. Save this raster as “LiDAR_data_processing\ACO\Snow_depth_processing\[Watershed]\Peripheral_masks\[Watershed]_[Year]_[Phase]_PeriMask.tif”. Note that if the noisy data falls outside the watershed boundary, masking is not necessary. If you see no noisy data, no action is required. If you see a lot of noisy throughout the watershed, it is possible that the LiDAR data needs to be reassessed and reprocessed by Hakai.

2.2 SnowDepth_corrections.py
This code clips out noisy areas (PeriMask), lakes, unrealistic negative values (<-5m), and noise introduced by vegetation misclassification or sensor issues. It also caps values >-5 & <0 to 0m and unrealistic high values to 10m + mean snowdepth. Noise introduced by vegetation misclassification or LiDAR sensor issues appear as ‘bumps’ or ‘throughs’ on the map. We clip out these bumps and throughs by applying a smoothing filter. You have to set the variables for this filter, which decide how aggressively it is applied:
-	Avalanche threshold. You want to avoid the removal of avalanches, which are ‘bumps’. Visually inspect the ortho data in QGIS to decide above which elevation avalanches appear. The filter is not applied above this elevation.
-	Upper and lower detection thresholds: the change in snowdepth that is assumed to be anomalous between neighboring areas
-	Kernel size: the size of the kernel smoothing moving average window
-	Expansion distance: the amount of pixels to be removed around an anomaly.  
The script outputs two processed snowdepth maps, one with and one without vegetation noise corrections. After visual inspection of both maps in QGIS, decide if, and with what settings you want to apply the vegetation noise corrections. Final smoothing variables are saved into “LiDAR_data_processing\ACO\Snow_depth_processing\[Watershed]\Processing_variables\”.

2.3 SnowDepth_gapfilling_interpolation.py & SnowDepth_gapfilling_modelling.py
These scripts fill gaps that were originally present in the LiDAR data or that were created by the noise corrections, but do not fill lake areas. The Cruickshank glaciers, however, are filled. Generally, small gaps are filled with interpolation, whereas larger gaps (such as glaciers, or large areas of vegetation misclassifications) are filled using modelling.

3 Snowdensity model generation
It is not mandatory to generate a new density model every time. If you do not need to update the density model, skip scripts 5 and 6 and go to the next section.

4 Snowdensity model inference
Snow density is estimated with a machine learning random forest model based on in-situ snowdensity & snowdepth, elevation (snow-free bare earth), slope, northness, eastness, curvature, canopy cover, canopy height, and Xt. The slope, northness, eastness, and curvature maps are calculated from elevation (snow-free bare earth), which can be done in QGIS: https://medium.com/@valentino.pintar/how-to-prepare-predictors-for-ecological-modelling-e442601120e3. Canopy cover and canopy height maps are prepared by Hakai. Xt will be prepared in the following step. 

4.1 SnowDensity_meteo_parameter_modelling
This script calculates Xt, a parameter used for snowdensity modelling. Xt is calculated from air temperature (positive degree days) and snowfall measured at one high and one low elevation meteorological station. The total precipitation data might or might not have to be QC’d yet, depending on the source of the meteo data. Uncomment the corrections that you want to apply (starting line 49). Note that the script assumes precip in mm and air temp in Celcius. You need to set an assumed rain/snow temperature threshold, which according to Jennings et al., 2018, is 0.97 for Metro Vancouver, 0.98 for Tsitika, 0.90 for Cruikshank, and 0.91 for Englishman watersheds.
This script outputs raster maps of Xt, Snowfall, and positive degree days and csv files of Xt, Snowfall, and positive degree days by elevation in “LiDAR_data_processing\ACO\Density_modelling\[Watershed]\Meteorological_parameter_modelling\[Year]\Output”. Currently, only Xt is used in the snow density model.

4.2 SnowDensity_model_inference
This script applies the machine learning random forest model to calculate snow density. The output is a snow density raster map, saved to “LiDAR_data_processing\ACO\Final_products\Maps\SnowDensity\[Watershed]\[Year]\[resolution]\[Watershed]_[Year]_[Phase]_SnowDensity.tif”.
If you do not have canopy data for your watershed (i.e., as for Metro Vancouver), use the “_NoCanpy” script. Otherwise, use the “_AllParameters” script. In the first code block (line 24-31), check that the right density model is being loaded. 
Make sure that all input rasters have the same extent, resolution, and project CRS. Because of computational limitations, we generally downscale all input maps to a 2m resolution and generate snow density maps at 2m resolution as well. The 0_prep.py script can perform the downscaling. 

5 SWE calculations
Snow water equivalent (SWE) is the product of snow depth and snow density. The following scripts calculate SWE, summary values, and uncertainties.

5.1 SWE_calculations
This script calculates and outputs a SWE raster map, and calculates subbasin summary values, including the mean snow depth (m), the mean snow depth above the snowline (m), the mean snow density (kg/m3), the mean SWE depth (mm), the total snow volume (m3), and the total snow water volume (m3). The SWE map is saved to “LiDAR_data_processing\ACO\Final_products\Maps\SWE\[Watershed]\[Year]\[resolution]\[Watershed]_[Year]_[Phase]_SWE.tif”, and the summary values are saved to “LiDAR_data_processing/ACO/SWE_calculations/[Watershed]/Key_numbers/[year]/[resolution]”.

5.2 SWE_elevation_banded_calculations
This script takes the SWE raster map(s) calculated in step 5.1 and calculates the mean SWE and total snow water volume by 100 m elevation bands. Results are added to “LiDAR_data_processing/ACO/SWE_calculations/[Watershed]/Elevation_banded_water_volumes/[year]/[resolution]/”.

5.3 SWE_uncertainty_propagation
This script calculates the percent and absolute uncertainty in total water volume (m3). The percent uncertainty can be applied to mean SWE to calculate the uncertainty in SWE. It is based, in part, on a random density model error (RDME), which is a value associated to the random forest model used for density calculations. The last recorded model runs by Rosie provided RDMEs of Cruikshank = 72, Englishman = 56, Metro Vancouver = 34, Tsitika = 51 kg/m3. For future updates of these models, the associated RDME will be saved in a file called ‘model_error_values.csv’ within the folder of that specific model (“LiDAR_data_processing\ACO\Density_modelling\[Watershed]\Overall_density_model”). Moreover, uncertainty is also based on errors in snow depth in areas below the snowline (i.e., pixel values deviating from 0 m). If the entire watershed is covered by snow, uncertainty calculations can not be performed and will be output as ‘0’ uncertainty.
Uncertainty is calculated based on the entire watershed. The same percent uncertainty is then applied to each subbasin. Uncertainty values are exported to “LiDAR_data_processing/ACO/SWE_calculations/[Watershed]/Key_numbers/[year]/[resolution]”.

6 Field checks and deliverables

6.1 Field_validation
This script compares snow depth, snow density, and SWE outputs to field observations that were made on the same day as (or only a few days removed from) the LiDAR survey. Multiple field measurements are taken at certain plots, usually measurements spaced about a meter apart in different cardinal directions. Generally, we assess plot averages, not single measurements. This script outputs csv files with plot-averaged statistics (i.e., mean and sd of field measurements in plot & mean and sd of LiDAR derived pixel values in plot) and ‘difference’ statistics (i.e., assessing the mean, sd, and rmse of differences (LiDAR – field).

6.2 Deliverables
This script gathers the various outputs generated throughout the workflow and merges them into summary tables. Under ‘date’, enter the date of today. All summary tables are output in “LiDAR_data_processing\ACO\Final_products\Tables\[Watershed]\[date]”. R scripts are available to generate summary figures from these summary tables, also available on the Github repository as “Figures_[Watershed].R”.
>>>>>>> 6b9dca538bc8e131366d018e1cc6da87936b6f14
