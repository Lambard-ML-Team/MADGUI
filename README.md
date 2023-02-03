# streamlit_app
User-friendly Graphical User Interface (GUI) developed at the National Institute for Materials Science (NIMS, MaDIS) for performing statistical data analysis, machine learning (ML) modelisation, and composition/process optimisation through Bayesian optimisation.

Can be used directly without any installation with the link below:
(add link when app deployed)

Developers:

![image](https://user-images.githubusercontent.com/108456770/215394202-2cba72ef-816d-4e05-af71-78343b82b6e4.png)

The application allow users to analyse their datas (with Pearson's correlation for example)

![image](https://user-images.githubusercontent.com/108456770/215407614-e26e4250-7864-495b-b0d4-e43f7efb4fb1.png)

In another section, the user has the ability to select the target, the prediction model, cross-validation and k-fold method for the prediction process.
We utilize four methods: ElasticNet, RandomForestRegressor, XGBRegressor or HistGradientBoostingRegressor, and two cross-validation methods: LeaveOneOut or K-fold.

![image](https://user-images.githubusercontent.com/108456770/215394425-03f5c70b-39cc-41a7-a8d4-4f9b0e6daf14.png)

An important part of the optimization is the limit for the features and the constraints, the application take both of it in consideration for the optimization.

![image](https://user-images.githubusercontent.com/108456770/215394498-d42ecad5-b5e2-4363-902d-f85e33951773.png)

The last part allow users to use prediction model and Bayesian Optimization at the same time to unbias the data for the initialisation of the Bayesian Optimization.

Here is an explanation of the Bayesian Optimization (source: http://krasserm.github.io/2018/03/21/bayesian-optimization/)

![image](https://user-images.githubusercontent.com/108456770/215394614-fa624138-568c-4951-b6e7-a4e9c3e005e0.png)

