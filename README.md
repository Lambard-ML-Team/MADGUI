<p align="center">
  <img src="https://user-images.githubusercontent.com/108456770/223624233-e40b55fa-fa50-4b37-a608-31077630311a.png" alt= “madgui_logo” width="200" height="200" margin:auto;">
</p>

# MADGUI : Material Design Graphical User Interface 

<p style="text-align:justify;">
User-friendly Graphical User Interface (GUI) developed at the National Institute for Materials Science (NIMS, MaDIS) for performing statistical data analysis, machine learning (ML) modelisation, and composition/process optimisation through Bayesian optimisation.
</p>

Can be used directly with the link below:

Streamlit app:  https://lambard-ml-team-madgui.streamlit.app/

Code accessible on GitHub:

GitHub page: https://github.com/Lambard-ML-Team/MADGUI

### Developers:

Christophe BAJAN* & Guillaume LAMBARD*

**National Institute for Materials Science, Tsukuba, Japan*


## Introduction

<p style="text-align:justify;">
We have developed MADGUI, a MAterial Design Graphical User Interface that require no programming knowledge and can be applied to a wide range of fields. This GUI is built using Python and various python libraries including Streamlit, scikit-learn, seaborn, xgboost and more importantly GpyOpt for the Bayesian Optimisation (BO) part. BO is a probability model that find the minimum/maximum of a black-box function (objective function) using a prior function based only on the data collected and performing multiple iterations. The goal of MADGUI is to help researchers to reach the optimum parameters in their research. 
</p>

The following parts are the explanation of how to use the GUI.
<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/223059434-bfa07661-1519-4b48-8a49-3d65c8e5623d.png" alt= “flowchart” width="800" height="450" margin:auto;">
</p>

## Data Preparation

Firstly, to use correctly the GUI you need to have a tabular dataset with some specifications. There is some rules to follow:
* Only numerical values
* No empty space
* Format csv or xlsx
* First line must be columns names (features and targets)
* Dataset must be in the first sheet of your file

Here is an exemple of what it must look like:

<img src="https://user-images.githubusercontent.com/108456770/229692238-a396619f-b0ac-4043-8d04-5316cf55c72b.png" alt= “dataset_exemple”>

When your file is prepared, you can use MADGUI by uploading your file via the button in the sidebar:

<img src="https://user-images.githubusercontent.com/108456770/229752477-9c94ea3c-9b9e-4aa0-8d7d-37b5940dc676.png" alt= “sidebar” width="167" height="459.5">

## Initialisation

<p style="text-align:justify;">
After uploading your data you have to select what columns are features and which one are the target. Take note that the columns where the standard deviation is 0 are already take out from the selection because it doesn't help the prediction or optimisation.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229695425-a3304dcb-614c-486a-acdb-e9ae968f6c92.png" alt = "selection_features" width="358.5" height="421" margin:auto;">
</p>

## Simple analysis

After your selection the GUI perform statistical analysis, first is a quick analysis of each columns (mean, min, max, std,...) and second is a correlation graph between each columns :

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229703230-bf4f9f3d-1ff6-4d6b-b610-cf4f10327114.png" alt = "quick_analysis" width="553" height="600"> <img src="https://user-images.githubusercontent.com/108456770/229703254-99e79a55-bdfe-4ece-88a3-86c8428c6e33.png" alt = "correlation" width="600" height="600">
</p>

The GUI also allow users to analyse their datas with Pearson's correlation, the Pearson correlation measures the strength of the linear relationship between two variables:

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229713726-eaf2009f-21cc-422c-8408-ec142b766477.png" alt = "Pearson" width="600" height="600" margin:auto;">
</p>

## Prediction

We utilize three machine learning methods: ElasticNet, RandomForestRegressor and XGBRegressor, with two kind of cross-validation: LeaveOneOut or K-fold for the prediction.

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229722250-cd608077-eb25-4594-b4ed-aa1052458d46.png" alt = "prediction_param" width="600" height="530"><img src="https://user-images.githubusercontent.com/108456770/229723072-d9ba65fb-a426-4cda-9ecb-c3456e9b59f7.png" alt = "prediction" width="600" height="600">
</p>

<p style="text-align:justify;">
After the prediction, you can use the feature importance graph to see which features have the highest score for the prediction model. If the model is accurate, you can then reduce the number of features needed by using only those that have scored high.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229725662-efce67b2-6240-4902-8438-80e52d7085eb.png" alt = "feature_importance" width="530" height="600">
</p>

## Optimisation

<p style="text-align:justify;">
An strength of this GUI is the possibilities to define features's limitation and to apply constraints, the application take both of it in consideration for the optimization.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229737188-2c60cb6b-3075-43aa-92d9-b1432d3039e3.png" alt = "limit_selection" width="530" height="600"><img src="https://user-images.githubusercontent.com/108456770/229737408-8663c8e2-be63-404b-ada3-364c8c791149.png" alt = "constraints" width="550" height="600">
</p>

<p style="text-align:justify;">
After all those limits and constraints you are ready to launch the optimisation. This GUI allow you to optimise one or several targets (up to 3), you can select for each one either to maximise or minimise it and also if you select multiple target you can determine the ratio between them (50-50 by default when two tagets are selected).
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229738442-40b938d3-5e2b-483b-92ea-5a5c8501b077.png" alt = "optimisation" width="550" height="600">
</p>

The result of the optimisation will be display has a dataframe and is downloadable as a csv file.

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229743191-2c970db1-93ec-4f2d-a6f2-4b219cf366af.png" alt = "optimisation_result" width="550" height="500">
</p>

The last part allow users to use prediction model and Bayesian Optimization at the same time to unbias the data for the initialisation of the Bayesian Optimization.

<p align="center">
<img src="https://user-images.githubusercontent.com/108456770/229746786-f4bfa5e8-1799-409a-9fc5-fbf038fd9620.png" alt = "optimisation_predi" width="550" height="600">
</p>

## Bayesian

Here is an explanation of the Bayesian Optimization (source: http://krasserm.github.io/2018/03/21/bayesian-optimization/)

<img src="https://user-images.githubusercontent.com/108456770/215394614-fa624138-568c-4951-b6e7-a4e9c3e005e0.png" alt="Bayesian_explication">

