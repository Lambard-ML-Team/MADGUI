<img src="https://user-images.githubusercontent.com/108456770/223624233-e40b55fa-fa50-4b37-a608-31077630311a.png" alt= “madgui_logo” width="200" height="200">

# MADGUI : Material Design Graphical User Interface 
User-friendly Graphical User Interface (GUI) developed at the National Institute for Materials Science (NIMS, MaDIS) for performing statistical data analysis, machine learning (ML) modelisation, and composition/process optimisation through Bayesian optimisation.

Can be used directly with the link below:

Streamlit app:  https://lambard-ml-team-madgui.streamlit.app/

Code accessible on GitHub:

GitHub page: https://github.com/Lambard-ML-Team/MADGUI

Developers:

Christophe BAJAN* & Guillaume LAMBARD*

**National Institute for Materials Science, Tsukuba, Japan*

<img src="https://user-images.githubusercontent.com/108456770/223059434-bfa07661-1519-4b48-8a49-3d65c8e5623d.png" alt= “flowchart” width="1000" height="600">

We have developed MADUI, a Material Design Graphical User Interface that require no programming knowledge and can be applied to a wide range of fields. This GUI is built using Python and various python libraries including Streamlit, scikit-learn, seaborn, xgboost and more importantly GpyOpt for the Bayesian Optimisation (BO) part. BO is a probability model that find the minimum/maximum of a black-box function (objective function) using a prior function based only on the data collected and performing multiple iterations. The goal of MADGUI is to help researchers to reach the optimum parameters in their research. 

The following parts are the explanation of how to use the GUI.

Firstly, to use correctly the GUI you need to have a tabular dataset with some specifications. There is some rules to follow:
* Only numerical values
* No empty space
* Format csv or xlsx
* First line must be columns names (features and targets)
* Dataset must be in the first sheet of your file

Here is an exemple of what it must look like:

![Screenshot 2023-04-04 at 14 07 37](https://user-images.githubusercontent.com/108456770/229692238-a396619f-b0ac-4043-8d04-5316cf55c72b.png)


The application allow users to analyse their datas (with Pearson's correlation for example)
We utilize four methods: ElasticNet, RandomForestRegressor, XGBRegressor or HistGradientBoostingRegressor, and two cross-validation methods: LeaveOneOut or K-fold.

![image](https://user-images.githubusercontent.com/108456770/223050037-8596b6bb-5d73-4d86-bed0-88a3eef6fbc7.png)

![image](https://user-images.githubusercontent.com/108456770/223052582-a2dc7cf4-a7f4-4123-9645-4eda6b172718.png)

An important part of the optimization is the limit for features and possibility to apply constraints, the application take both of it in consideration for the optimization.
The last part allow users to use prediction model and Bayesian Optimization at the same time to unbias the data for the initialisation of the Bayesian Optimization.

![image](https://user-images.githubusercontent.com/108456770/223050417-fe3c8d5b-636d-4176-bedd-ede96b9ff371.png)

Here is an explanation of the Bayesian Optimization (source: http://krasserm.github.io/2018/03/21/bayesian-optimization/)

![image](https://user-images.githubusercontent.com/108456770/215394614-fa624138-568c-4951-b6e7-a4e9c3e005e0.png)

