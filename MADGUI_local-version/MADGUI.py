# To run this file tap: ' streamlit run MADGUI.py  ' in the terminal while being in the repertory of this file.
# The recessary librairy are :

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import glob as gb
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Whisker

from re import search

import sklearn as skl
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict, cross_validate, KFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor

import GPyOpt as gpopt

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib import font_manager as fm


import os
import io
from pathlib import Path

import seaborn as sns
from scipy.optimize import minimize


# Font for Japanese character in matplotlib and seaborn

fpath = os.path.join(os.getcwd(), "Noto_Sans_JP/NotoSansJP-Regular.otf")
prop = fm.FontProperties(fname=fpath)
font_dir = ['Noto_Sans_JP']
for font in fm.findSystemFonts(font_dir):
    fm.fontManager.addfont(font)
rcParams['font.family'] = 'Noto Sans JP'

# First define the different part that we will display

header = st.container()
dataset = st.container()
feature_selection = st.container()
correlation = st.container()
prediction = st.container()
limite = st.container()
bayesian = st.container()

# Tool for activate the cursor on the bokeh graph to display the value
TOOLTIPS = [("index", "$index"),("(x,y)", "(@x, @y)"),]

# Function to covert data to csv
@st.cache_data
def convert_feat_lim(df):
	return df.to_csv().encode('utf-8')

	############
	### In this program we have different pages. To be able to call variable through different page
	### we use the st.session_state['variable'] = variable
	### Then we call them back in the other pages by doing the opposite :
	### variable = st.session_state['variable']
	############

# The navigation between pages will be on the sidebar with the data uploader and the reset button

with st.sidebar:
	st.image('madgui_logo.png',use_column_width='auto')
	# Navigation part in the sidebar
	choice = option_menu('Navigation', ['Main Page','Prediction','Bayesian','About'],
		icons = ['house', 'tree','app-indicator','info-circle'],
		menu_icon = 'map', default_index=0,
		styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"}})

	# Upload data part
	uploaded_file = st.file_uploader('Upload your data', type = ['.csv','.xlsx'])
	
	# Reset button part
	reset = st.sidebar.button('Reset all')
	if reset:
		st.session_state = {}
		uploaded_file = {}
	

if choice == 'Main Page':
	
	############
	### On this page the user must load his data, select his feature and his target 
	### and check if everything is correct before continuing
	############

	with header: 
		# This section is to explain briefly what the point of the app

		st.title('MADGUI - Data Analysis using prediction and Bayesian Optimisation')
		st.write('Welcome! The objective of this project is to help you to analyse your data, to find the best \n'
			'next sample to reach your objective using Bayesian Optimisation.\n'
			'You will be asked to complete different parts of this program. You must start by completing the Main Page where you are currently located. Then you will be able to do either the prediction or the Bayesian optimisation. \n'
			"Take note that if you change anything after submitting your selection, you must click 'Submit' again.")
		
		st.header('1 - Data upload')
		st.info("Read carefully before uploading your data ")


		# Explanation of the rules for uploading the data with example
		
		with st.expander('⚠ Read before continuing ⚠'):
			st.write('Before uploading, you have to make sure that your data (csv/xlsx) look like this :')
			st.image('Data/exemple_data.png')
			st.write('The first line of your data file must contain the names of your columns. Your data should be in the first sheet of your file and there should be no blank cells. All values in your dataset should be numerical.')
			st.write("Features are the parameters that you can change during your experiment. Targets are the results of those experiment and are your objective.")
			st.write("For instance, Feature_1 and Feature_4 are numerical values, while Feature_2 and Feature_3 are categorical values (respectively 2 and 3 choices). Feature_2 can be 'with' or 'without' something  where 'with' is represented by the numerical value of 1 and 'without' is represented by the numerical value of 2.\n"
				'It is important that there are no blank cells in your data and that all values used are in numerical format.')
			st.write('After uploading you should see something like this :')
			exemple = pd.read_csv('Data/Exemple_data.csv',sep = ',')
			st.dataframe(exemple)
			st.subheader('Upload your dataset on the sidebar')
			
		st.write('Firstly, you will have to upload your data. To do that you have to click on the "Browse File" button in the sidebar.')
	
	with dataset: 
		# This part is to read the data with pandas as a DataFrame and display the selected data to show if it worked correctly

		if not uploaded_file:
			st.stop()
			
		if uploaded_file.name[-4:] == '.csv':
			data_file = pd.read_csv(uploaded_file)
		elif uploaded_file.name[-5:] == '.xlsx':
			data_file = pd.read_excel(uploaded_file, sheet_name = 0)

		st.session_state['data_file'] = data_file

		st.write(f'Your data from the file {uploaded_file.name} are represented below, you can check if everything is alright.')
		st.dataframe(data_file)
			
	with feature_selection: 
		# This part is very important, it is the part where the user select which
		# columns of his data are feature and which are target.

		st.header('2 - Selection of your features and targets for the project')
		st.write('On this section of the program, you must select which columns of your dataset are the features you want to analyse and which columns are the targets that you want to predict or improve.')
		
		### Selection of the feature
		st.write('Columns with a standard deviation of 0 are already deselected, as well as columns that contain text. This is done automatically to eliminate columns that do not provide useful information for analysis and prediction.')

		# Select the columns where the standard deviation is different than 0
		numeric_cols = st.session_state['data_file'].select_dtypes(['number'])
		test = np.std(numeric_cols)
		list_column = np.where(test!=0)
		test2=[]
		for i in list_column:
			test3 = test.axes[0][i]
			test2 = np.hstack((test2,test3))

		# Form with the selection of feature and target
		with st.form(key='my_form'):

			if 'feature_selected' not in st.session_state:
				feature = st.multiselect("Features - Unselected the one you don't need :", st.session_state['data_file'].columns, default = test2)
			else:
				feature = st.multiselect("Features - Unselected the one you don't need :", st.session_state['data_file'].columns, default = st.session_state['feature_selected'])

			if 'target_selected' not in st.session_state:
				target = st.multiselect('Select your targets then click the submit button :', data_file.columns)
			else:
				target = st.multiselect('Select your targets then click the submit button :', data_file.columns, default = st.session_state['target_selected'])

			submit_button = st.form_submit_button(label='Submit')
						
		if submit_button:
			# Save the selection into safe variable to be able to use them in the other pages
			data_file_feature = data_file[[feature[i] for i in range(len(feature))]]
			st.session_state['data_file_feature'] = data_file_feature
			st.session_state['feature_selected'] = feature
			st.session_state['target_selected'] = target
			data_file_target = data_file[[target[i] for i in range(len(target))]]
			st.session_state['data_file_target'] = data_file_target
			st.session_state["Check1"] = True
			st.session_state["Check_lim"] = False
			st.session_state["constraints"]={}

		if "Check1" in st.session_state:
			st.write('The data that you selected (Features + Targets) are : ')
			data_file_selected = pd.concat([st.session_state['data_file_feature'], st.session_state['data_file_target']], axis = 1)
			st.session_state['data_file_selected'] = data_file_selected
			st.dataframe(data_file_selected)

			# Quick analyse of the data selected using pd.describe
			st.header('3 - Quick analysis')
			st.write('Here you can see some information about your data :')
			data_describe = st.session_state['data_file_selected'].describe(include = 'all')
			st.dataframe(data_describe)

			lim_feature = len(st.session_state['feature_selected'])
			st.session_state['lim_feature'] = lim_feature
			# Bar chart that show the target value of each target
			X = pd.DataFrame(np.arange(len(st.session_state['data_file_selected']))+1)
			Z=X

			for i in range(len(st.session_state['data_file_target'].axes[1])):
				Y = pd.concat([X, st.session_state['data_file_target'].iloc[:,i]],axis=1)
				st.bar_chart(data=Y.iloc[:,1], y=Y.axes[1][0])
				Z = pd.concat([Z, st.session_state['data_file_target'].iloc[:,i]],axis=1)

			# Graph with the correlation between each features and target 
			st.write("By clicking on the button below, you will display the correlation graph between all the features/target that you selected. It can take some times to charge it.")
			correlation = st.button("Display the correlation graph")
			
			if correlation:
				corr_fig = sns.pairplot(st.session_state['data_file_selected'].iloc[:,:],corner=True)
				st.pyplot(corr_fig)
				fn_2 = 'correlation.png'
				img = io.BytesIO()
				plt.savefig(img, format='png')
				btn = st.download_button(
					label='Download graph',
					data=img,
					file_name=fn_2,
					mime='image/png')
			st.success('You can now navigate to the other pages of the program. It is important to note that if you make any changes to your selection on this page, you must submit it again to ensure that the changes are saved and applied to the analysis.')
			
			# To initialize this variable without it being reset everytime in the next page
			st.session_state['test']=False #For the third page (constraints)
			st.session_state['check'] = 0
		else:
			st.info('Submit your selection.')
			st.stop()
		
elif choice == 'Prediction':

	############
	### On this page the user can see the Pearson correlation between his variable and also the user  
	### can use ElasticNet, RandomForest, XGBRegressor to predict the value of a specific target
	############

	# Part to tell the user to complete the Main Page before coming here.
	if 'lim_feature' not in st.session_state:
		st.warning("Please upload your data in 'Main Page' and select your feature")
		st.stop()

	# Here it is to load the variable define in Main page, we can just call the variable  
	# to use it in different pages, we need to first save it in st.session_state then call it
	data_file_feature = st.session_state['data_file_feature']
	data_file_target = st.session_state['data_file_target']
	data_file_selected = st.session_state['data_file_selected']
	lim_feature = st.session_state['lim_feature']
	feature_selected = st.session_state['feature_selected']
	target_selected = st.session_state['target_selected']

	st.session_state['data_file_feature'] = st.session_state['data_file_selected'][[st.session_state['feature_selected'][i] for i in range(len(st.session_state['feature_selected']))]]

	with correlation:
		# Pearson correlation using the features and targets selected in Main page
		st.title('Prediction')
		st.header('Pearson linear correlation visualisation')
		st.write("The Pearson correlation measures the strength of the linear relationship between two variables.")
		
		st.session_state['data_selected'] = pd.concat([st.session_state['data_file_feature'].iloc[:],st.session_state['data_file_target']],axis=1)
		
		data_corr = st.session_state['data_selected'].iloc[:,:].corr(method = 'pearson')
		data_corr[(data_corr <= 0.5) & (data_corr >= -0.5)] = np.nan
		fig, ax = plt.subplots(figsize=(15,15)) 
		data_corr_mask = np.triu(data_corr)
		sns.heatmap(data_corr, 
		            xticklabels = st.session_state['data_selected'].iloc[:,:].columns,
		            yticklabels = st.session_state['data_selected'].iloc[:,:].columns, 
		            linewidths = 0.1,
		            vmin = -1, 
		            vmax = 1, 
		            annot = True, 
		            ax = ax, 
		            cmap = "coolwarm",
			    mask = data_corr_mask);
		st.pyplot(fig)
		fn = 'pearson_corr.png'
		img = io.BytesIO()
		plt.savefig(img, format='png')
		btn = st.download_button(
			label='Download graph',
			data=img,
			file_name=fn,
			mime='image/png')

	with prediction:

		st.header("Target's Prediction using different methods (ElasticNet, RandomForest and XGBRegressor)")

		with st.expander('Explanation'):
			st.write('Here you will have to choose between different methods of prediction: ElasticNet, RandomForest or XGBRegressor.'
				'\n\n You will also have to choose the cross validation that you want to use. '
				"Cross-validation is a method used in machine learning to assess the performance of a model. \n\n K-fold cross-validation and Leave-One-Out cross-validation are two different methods of cross-validation. The main difference between the two is how the data is divided into subsets for training and validation. K-fold cross-validation divides the data into K equally sized folds and repeat the process k times, while Leave-One-Out cross-validation uses all but one data point as the training set and repeat the process N times where N is the number of data points in the dataset.")

		# This function is the function to use the predictor (ElasticNet, RandomForest, XGBRegressor), it can certainly be move outside of this 
		# file and then be called here.

		def analyse_function(method, data, target, crossval, lim_feature, k_num=3,random_state = 0, n_estimators = 100):

			regressors = {'ElasticNet': ElasticNet(random_state=0),
              'RandomForestRegressor': RandomForestRegressor(n_estimators=n_estimators, random_state=0),
              'XGBRegressor': XGBRegressor(n_estimators=n_estimators, seed=0)}
#               'HistGradientBoostingRegressor': HistGradientBoostingRegressor()}
			    
			def cross_val_est_fn(clf, x, y, cv):
				predictions = cross_val_predict(estimator = clf, 
				                                X = x, 
				                                y = y, 
				                                cv = cv)
				validation_arrays = cross_validate(estimator = clf, 
				                                  X = x, 
				                                  y = y, 
				                                  cv = cv, 
				                                  scoring = scorer, 
				                                  return_estimator = True)   

				test_mae, test_mse, estimator = validation_arrays['test_mae'], validation_arrays['test_mse'], validation_arrays['estimator']

				return predictions, -test_mae, np.sqrt(-test_mse), estimator
		    
		    #Function to plot the feature importance
			def features_importance_plot(x, features_importance_mean, feature_importances_sd,method,target):

			    fig, ax = plt.subplots(figsize = (10,5))
			    nx = np.arange(x.shape[1])
			    ax.bar(nx, features_importance_mean, xerr = 0, yerr = feature_importances_sd, align = 'center')
			    ax.set_xticks(nx)
			    ax.set_xticklabels(features_importance_mean.keys().values.tolist(), rotation = 'vertical')
			    ax.tick_params(labelsize = 14)
			    ax.set_title("Features importance for %s"%target + " with %s"%method, fontsize = 18);
			    
			    return fig

			# the target selected by the user
			feature_columns = np.where(data.columns == target)[0][0]
			regressor = regressors[method]

			clf = make_pipeline(StandardScaler(), regressor)

			if crossval=='LeaveOneOut':
			    crossvalidation = LeaveOneOut()
			elif crossval=='K-Fold':
			    crossvalidation = KFold(n_splits=k_num, shuffle=True, random_state=0)

			mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
			mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
			scorer = {'mae': mae_scorer, 'mse': mse_scorer}

			# Function to determine the feature importance in the prediction
			def features_importance_fn(x, estimator):
				col_name = x.columns.tolist()
				for idx, estimator in enumerate(estimator):
					if method == 'ElasticNet':
					    if idx == 0:
					        feature_importances = pd.DataFrame(estimator[1].coef_, index = col_name, 
					                                           columns=['importance'+str(idx)])
					    else: 
					        feature_importances['importance'+str(idx)] = estimator[1].coef_
					else:
						if idx == 0:
							feature_importances = pd.DataFrame(estimator[1].feature_importances_,
				                                           index = col_name,
				                                           columns = ['importance'+str(idx)])
						else: 
							feature_importances['importance'+str(idx)] = estimator[1].feature_importances_

				feature_importances_mean = feature_importances.mean(axis = 1).sort_values(ascending = False)
				feature_importances_sd = feature_importances.std(axis = 1).loc[feature_importances_mean.keys().values.tolist()]

				return feature_importances_mean, feature_importances_sd

			# Prediction with cross validation using LeaveOneOut or K-fold and the method selected by user
			pred_, test_mae_, test_rmse_, est_ = cross_val_est_fn(clf = clf, x = data.iloc[:,0:lim_feature], 
			                                                      y = data.iloc[:,feature_columns], 
			                                                      cv = crossvalidation)
			fig = figure(
				title='Prediction using %s'%method +' versus Observation',
				sizing_mode="stretch_width",
				x_axis_label='Observed ' + target,
				y_axis_label='Predicted ' + target,
				tooltips=TOOLTIPS,
				)

			fig.circle(data.iloc[:,feature_columns], pred_)
			fig.line([np.floor(np.min(data.iloc[:,feature_columns])*10)/10., np.ceil(np.max(data.iloc[:,feature_columns])*10)/10.],
			         [np.floor(np.min(data.iloc[:,feature_columns])*10)/10., np.ceil(np.max(data.iloc[:,feature_columns])*10)/10.], 
			         line_color="orange", line_dash="4 4")

			# plot of the features importance
			fi_feature_mean, fi_feature_sd = features_importance_fn(data.iloc[:,0:lim_feature], estimator = est_)
			fig2 = features_importance_plot(data.iloc[:,0:lim_feature], fi_feature_mean, fi_feature_sd, method,target)

			return pred_, fig, fig2, test_mae_, test_rmse_, est_

		methods = ['ElasticNet', 'RandomForestRegressor', 'XGBRegressor']
		crossval_list = ['LeaveOneOut','K-Fold']
		
		with st.form('Prediction'):

			target = st.selectbox('Select which target you want to predict',st.session_state['target_selected'])
			method = st.selectbox('Select which method of prediction you want to use',methods)
			crossval = st.selectbox('Select which method of cross validation you want to use',crossval_list)
			k_num = st.number_input('Choose how many subsets do you want to use, it has an impact only if you selected K-fold',min_value=2,max_value=len(st.session_state['data_file_selected'].iloc[:,0])-1,value=3)

			predict = st.form_submit_button('Submit')	

		# Only calculate if button is click

		if predict:

			st.session_state['target_predi'] = target
			st.session_state['method'] = method
			st.session_state['crossval'] = crossval
			st.session_state['k_num'] = k_num

			pred_, fig, fig2, test_mae_, test_rmse_, est_ = analyse_function(method = method, 
										data = st.session_state['data_selected'], 
										target = target, 
										crossval=crossval,
										k_num=k_num,
										lim_feature = lim_feature)
			st.session_state['pred_']=pred_
			st.session_state['fig']=fig
			st.session_state['fig2']=fig2
			st.session_state['test_mae_']=test_mae_
			st.session_state['test_rmse_']=test_rmse_
			st.session_state['est_S'] = est_

		# Part to add the buttons to save the graph
		if 'fig' in st.session_state:

			# MAE and RMSE:
			if st.session_state['crossval'] == 'LeaveOneOut':
				st.write('Non-linear regression on %s'%st.session_state['target_predi'] + ' with %s'%st.session_state['method'] + ' and %s'%st.session_state['crossval'] + ' :\n\tMAE: {0:.3f} +/- {1:.3f}\n\tRMSE: {2:.3f} +/- {3:.3f}'.\
			    	format(np.mean(st.session_state['test_mae_']), np.std(st.session_state['test_mae_']), np.mean(st.session_state['test_rmse_']), np.std(st.session_state['test_rmse_'])))
			if st.session_state['crossval'] == 'K-Fold':
				st.write('Non-linear regression on %s'%st.session_state['target_predi'] + ' with %s'%st.session_state['method'] + ' and %s'%st.session_state['crossval'] + ' = %s'%st.session_state['k_num'] + ' :\n\tMAE: {0:.3f} +/- {1:.3f}\n\tRMSE: {2:.3f} +/- {3:.3f}'.\
			    	format(np.mean(st.session_state['test_mae_']), np.std(st.session_state['test_mae_']), np.mean(st.session_state['test_rmse_']), np.std(st.session_state['test_rmse_'])))
			
			# Graphic about the prediction over the observation
			st.bokeh_chart(st.session_state['fig'])

			# Preparation of the data for the csv export
			feature_columns = np.where(data_file_selected.columns == target)[0][0]
			prediction_data = pd.DataFrame(np.zeros((len(data_file_selected),2)))
			prediction_data.columns=['%s'%target,'Prediction using %s'%method]
			prediction_data.iloc[:,0]=data_file_selected.iloc[:,feature_columns]
			prediction_data.iloc[:,1]=st.session_state['pred_']
			prediction_data_csv = convert_feat_lim(prediction_data)
			btn = st.download_button(
				label='Download Data (CSV)',
				data=prediction_data_csv,
				file_name='data_prediction.csv',
				mime='text/csv')

			# Graphic about the difference between the observation and the prediction
			fig_diff = figure(
		    	title='Difference between observation and prediction',
		    	sizing_mode="stretch_width",
    			x_axis_label='Samples',
    			y_axis_label='Obs - Pred',
    			tooltips=TOOLTIPS,
				)			
			sample = pd.DataFrame(np.arange(len(st.session_state['data_file_selected'])))
			feature_columns = np.where(data_file_selected.columns == target)[0][0]
			diff_pred = data_file_selected.iloc[:,feature_columns]-st.session_state['pred_']
			fig_diff.circle(sample[0],diff_pred)
			fig_diff.line(sample[0],np.zeros(len(sample[0])),line_color="orange", line_dash="4 4")
			st.bokeh_chart(fig_diff)

			# Graphic about the difference between the gap Obs-Pred and the Observation point

			fig_diff2 = figure(
		    	title='Difference between observation-prediction and Observation',
		    	sizing_mode="stretch_width",
    			x_axis_label='Observed ' + target,
    			y_axis_label='Obs - Pred',
    			tooltips=TOOLTIPS,
				)			
			feature_columns = np.where(data_file_selected.columns == target)[0][0]
			diff_pred = data_file_selected.iloc[:,feature_columns]-st.session_state['pred_']
			fig_diff2.circle(data_file_selected.iloc[:,feature_columns],diff_pred)
			fig_diff2.line(data_file_selected.iloc[:,feature_columns],np.zeros(len(sample[0])),line_color="orange", line_dash="4 10")
			st.bokeh_chart(fig_diff2)

			# Graphic about the features importance
			
			st.pyplot(st.session_state['fig2'])
			st.session_state['img2'] = io.BytesIO()
			fn3 = 'feature_importance.png'
			plt.savefig(st.session_state['img2'],dpi='figure',format='png',bbox_inches='tight')
			btn = st.download_button(
				label='Download graph',
				data=st.session_state['img2'],
				file_name=fn3,
				mime='image/png')
	
			# Part about the possibility to reselect your feature with the help of the graph about the feature importance
			
			with st.form(key='second_selection'):
				st.write("Here you can choose to change your feature's selection depending on the result of the features importance graph.")
				feature_2nd_selec = st.multiselect("Features - Unselected the one you don't need then click the submit button :", st.session_state['data_file'].columns, default = st.session_state['feature_selected'])
				submit_2nd_selec = st.form_submit_button('Submit')

			if submit_2nd_selec:
				st.session_state['feature_selected']=feature_2nd_selec
				lim_feature = len(st.session_state['feature_selected'])
				st.session_state['lim_feature']=lim_feature
				data_file_feature = st.session_state['data_file_selected'][[st.session_state['feature_selected'][i] for i in range(len(st.session_state['feature_selected']))]]
				st.session_state['data_file_feature'] = data_file_feature
				st.session_state['data_selected'] = pd.concat([st.session_state['data_file_feature'].iloc[:],st.session_state['data_file_target']],axis=1)
				st.dataframe(st.session_state['data_selected'].iloc[:,0:lim_feature])
				st.experimental_rerun()

		# Determination of the best parameter for prediction, determined by finding the minimum value of the sum of Observation-prediction
		st.subheader('Best prediction parameters')
		st.write("The button below will perform every possible combinaison for the prediction of the target that you select above and return the best one.\n\n"
			"Be careful that it can take a lot of time (severals minute) depending on the number of data that you have.")


		def analyse_function_pred(method, data, target, crossval, lim_feature, k_num=3,random_state = 0, n_estimators = 100):
	    
		    if method == 'ElasticNet':
		        regressor = ElasticNet(random_state = 0)
		    elif method == 'RandomForestRegressor':
		        regressor = RandomForestRegressor(n_estimators = n_estimators, random_state = 0)
		    elif method == 'XGBRegressor':
		        regressor = XGBRegressor(n_estimators = n_estimators, seed = 0)		        
		        
		    def cross_val_est_fn(clf, x, y, cv):
		        predictions = cross_val_predict(estimator = clf, 
		                                        X = x, 
		                                        y = y, 
		                                        cv = cv)
		        return predictions
		    
		    # the target selected by the user
		    feature_columns = np.where(data.columns == target)[0][0]

		    clf = make_pipeline(StandardScaler(), regressor)

		    if crossval=='LeaveOneOut':
		        crossvalidation = LeaveOneOut()
		    elif crossval=='K-Fold':
		        crossvalidation = KFold(n_splits=k_num, shuffle=True, random_state=0)

		    # Prediction with cross validation using LeaveOneOut or K-fold and the method selected by user
		    pred_ = cross_val_est_fn(clf = clf, x = data.iloc[:,0:lim_feature], 
		                                                          y = data.iloc[:,feature_columns], 
		                                                          cv = crossvalidation)
		    return pred_
		best_pred = st.button('Find the best parameters')
		methods = ['ElasticNet', 'RandomForestRegressor', 'XGBRegressor']
		crossvalidation = ['LeaveOneOut','K-Fold']
		if best_pred:
			analyse_pred=[]
			indice=[]
			for i in methods:
				for j in crossvalidation:
					if j=='LeaveOneOut':
						k=0
						pred_ = analyse_function_pred(method = i, 
													data = st.session_state['data_selected'], 
													target = target, 
													crossval=j,
													k_num=0,
													lim_feature = lim_feature)
						row=np.array([np.sum(np.abs(st.session_state['data_file_selected'].iloc[:,feature_columns]-pred_))])
						analyse_pred.append(row)
						indice.append([i,j,k])
					else:
						for k in range(2,len(st.session_state['data_selected'].iloc[:,0])-1):
							pred_ = analyse_function_pred(method = i, 
													data = st.session_state['data_selected'], 
													target = target, 
													crossval=j,
													k_num=k,
													lim_feature = lim_feature)
							row=np.array([np.sum(np.abs(st.session_state['data_file_selected'].iloc[:,feature_columns]-pred_))])
							analyse_pred.append(row)
							indice.append([i,j,k])

			st.session_state['analyse_pred'] = np.array(analyse_pred)
			st.session_state["indice"] = indice

		if 'analyse_pred' in st.session_state:		
			a = np.argmin(st.session_state['analyse_pred'][:].astype(np.float))
			st.dataframe(st.session_state["indice"][a])


			_,fig,_,test_mae_,test_rmse_,_= analyse_function(method = st.session_state["indice"][a][0], 
											data = st.session_state['data_selected'], 
											target = target, 
											crossval= st.session_state["indice"][a][1],
											k_num=st.session_state["indice"][a][2],
											lim_feature = lim_feature)
						# MAE and RMSE:
			if st.session_state["indice"][a][1] == 'LeaveOneOut':
				st.write('Non-linear regression on %s'%st.session_state['target_predi'] + ' with %s'%st.session_state["indice"][a][0] + ' and %s'%st.session_state["indice"][a][1] + ' :\n\tMAE: {0:.3f} +/- {1:.3f}\n\tRMSE: {2:.3f} +/- {3:.3f}'.\
			    	format(np.mean(test_mae_), np.std(test_mae_), np.mean(test_rmse_), np.std(test_rmse_)))
			if st.session_state["indice"][a][1] == 'K-Fold':
				st.write('Non-linear regression on %s'%st.session_state['target_predi'] + ' with %s'%st.session_state["indice"][a][0] + ' and %s'%st.session_state["indice"][a][1] + ' = %s'%st.session_state["indice"][a][2] + ' :\n\tMAE: {0:.3f} +/- {1:.3f}\n\tRMSE: {2:.3f} +/- {3:.3f}'.\
			    	format(np.mean(test_mae_), np.std(test_mae_), np.mean(test_rmse_), np.std(test_rmse_)))
			

			st.bokeh_chart(fig)
		
elif choice == 'Bayesian':

	############
	### On this page the user must choose the limitation and step of all his feature 
	### then the program use bayesian optimisation to give him as much new sample as the user choose with the slider
	############

	if 'lim_feature' not in st.session_state:
		st.warning("Please upload your data in 'Main Page' and select your feature")
		st.stop()

	# Here it is to load the variable define in Main page, we can just call the variable  
	# to use it in different pages, we need to first save it in st.session_state then call it
	data_file_feature = st.session_state['data_file_feature']
	data_file_target = st.session_state['data_file_target']
	data_file_selected = st.session_state['data_file_selected']
	lim_feature = st.session_state['lim_feature']
	feature_selected = st.session_state['feature_selected']
	target_selected = st.session_state['target_selected']

	with limite:

		# In this part we need to ask the user to complete a csv file with the limits of his features 
		# as well as his constraints, to be able to perform the Bayesian Optimisation

		st.title('Limit Selection')
		st.write('In this part, you will have to complete the dataframe below with the min, \n'
		' max and step values or the specific values of each features that you have previously selected. \n'
		'\n\nTo do this, you will need to download the CSV file generated below, complete it and then upload it.')
		st.write("If you have a feature that can takes only some specific values, you can write them in the column 'Specific values' and write them as shown below. For example: 1, 3.4 ,5.6 ,13. ")
		st.write('Additionally, if you want a feature to have only one specific value, you can write it in the Specific Values column like this: 100,100 or 0,0 for example if you want the value to be 100 or 0.')
		
		# The dataframe that the user must complete

		
		# Make a template of the file that the user must complete with an example about the specific values
		data_lim_example = pd.DataFrame(np.zeros((len(st.session_state['feature_selected']),4)),
			columns = ('Min','Max','Step','Specific Values'), 
			index = (st.session_state['feature_selected']))
		data_lim_example.iloc[0,3]='1,3.4,5.6,13'
		data_lim_example.iloc[1,3]='100,100'

		st.dataframe(data_lim_example)
		# Allow the user to download the file about the feature's limits that the user must complete

		test = np.array(np.zeros((len(st.session_state['feature_selected']),1)),dtype='U')
		test[:]="0"
		df = np.array(np.concatenate((np.zeros((len(st.session_state['feature_selected']),3)),test),axis=1))
		data_lim = pd.DataFrame(df,
			columns = ('Min','Max','Step','Specific Values'), 
			index = (st.session_state['feature_selected']))
		data_lim_csv = convert_feat_lim(data_lim)
		st.write('Click below to download a CSV. Fill out the template with the boundaries and step of the features you selected before. Then you will have to upload it below')
		st.download_button(
			label = 'Click here to download the file with your features',
			data = data_lim_csv,
			file_name = 'limite_feature.csv',
			mime = 'text/csv')
		
		st.header('Bayesian Optimisation')
		st.subheader('Limits and constraints')
		
		column_lim1 , column_lim2 = st.columns(2)
		with column_lim1: # interactive dataframe
			with st.form(key='Limit'):
				new_data_lim = st.experimental_data_editor(data_lim)
				validate_lim = st.form_submit_button(label='Validate limits')
			if validate_lim:
				st.session_state['feature_lim'] = new_data_lim
				st.session_state["Check_lim"] = True
				st.session_state["check_upload"]=False
				data_lim_csv = convert_feat_lim(st.session_state['feature_lim'])		
				st.download_button(
					label = 'Save your limits for next time',
					data = data_lim_csv,
					file_name = "lim_feature.csv",
					mime = 'text/csv')
		with column_lim2: # upload the csv file
			st.subheader('Upload the file containing the limite that you fixed')
			def on_file_upload():
				# This part is useful in case the user uploaded the wrong file or want to changed
				st.session_state['check'] +=1
			# The user must submit the file with the limits here
			uploaded_lim = st.file_uploader('Select the file that you have completed', type = ['.csv'],on_change=on_file_upload)
			if uploaded_lim:
				st.session_state["check_upload"]=True
				feature_lim = pd.read_csv(uploaded_lim)
				st.session_state['feature_lim'] = pd.DataFrame(feature_lim.iloc[:,1:].values,
				columns = ('Min','Max','Step','Specific Values'), 
				index = (st.session_state['feature_selected']))
				st.session_state['uploaded_lim'] = uploaded_lim
		if 'check_upload' not in st.session_state or 'Check_lim' not in st.session_state:
			st.stop()
		if validate_lim:
			st.session_state['feature_lim'] = new_data_lim
			st.session_state["check_upload"]=False
			st.session_state["Check_lim"]=True
		elif st.session_state["check_upload"]:
			validate_lim = False	

	with bayesian:

		st.markdown('_Make sure that everything is correct_')

		# If the file uploaded doesn't have the same number of line than the file downloaded,
		# tell the user that there is a mistake
		if len(data_lim)!= len(st.session_state['feature_lim']):
			st.error("The file that you uploaded doesn't have the same number of line than your selection, make sure that you uploaded the right file or check your feature selection.")
		
		# Space of paramaters to be evaluated

		st.write('With the limits that you choose, the possible values for each features are :')
		space = []
		
		for i in range(len(st.session_state['feature_lim'].iloc[:,0])):
			if st.session_state['feature_lim'].iloc[i,3] == 0:
				row = [{'name': st.session_state['feature_lim'].index[i], 'type': 'discrete', 'domain': [it for it in np.arange(float(st.session_state['feature_lim'].iloc[i,0]), float(st.session_state['feature_lim'].iloc[i,1]) + float(st.session_state['feature_lim'].iloc[i,2]), float(st.session_state['feature_lim'].iloc[i,2]),dtype=float)]}]
				space.extend(row)
			elif st.session_state['feature_lim'].iloc[i,3] == '0':
				row = [{'name': st.session_state['feature_lim'].index[i], 'type': 'discrete', 'domain': [it for it in np.arange(float(st.session_state['feature_lim'].iloc[i,0]), float(st.session_state['feature_lim'].iloc[i,1]) + float(st.session_state['feature_lim'].iloc[i,2]), float(st.session_state['feature_lim'].iloc[i,2]),dtype=float)]}]
				space.extend(row)
			else:
				row = [{'name': st.session_state['feature_lim'].index[i], 'type': 'discrete', 'domain': [it for it in np.array(np.fromstring(st.session_state['feature_lim'].iloc[i,3], dtype=float, sep=','))]}]
				space.extend(row)
		
		st.dataframe(space)
		
		
		#Number of possibilities:
		st.write("The number of possibilities are :", "{:.2e}".format(np.prod([len(space[irow]['domain']) for irow in range(len(space))])))

		# Explaination about consraints

		st.subheader("Constraints")
		st.write("In this section, you are prompted to input any constraints that may apply to your analysis.")
		st.write('The equation should be of the form : x[:,i] + x[:,j] = 1 where i and j correspond to the line of the feature in the Dataframe above and 1 is your restriction. \n'
		'It can also be of the form x[:,i] + x[:,j] ∈ [0.5,1.2] to say that i and j must be inside the array [0.5,1.2].'
		' You will need to adjust the values of i and j and the last number (if necessary).\n'
		'\n\n  Exemple : If you want x[:,i] + x[:,j] = 1, you must write:  x[:,i] + x[:,j] in the first space and 1 in the other.\n'
		'\n\n  Exemple 2 : If you want x[:,i] + x[:,j] ∈ [0.5,10], you must write:  x[:,i] + x[:,j] in the first space and [0.5,10] in the other. \n\n'
		'It is recommended that you write down the constraint that you put here so you can remember them correctly later if you want to perform the same analysis.')

		# Selection of how many constraints the user want
		column1, column2 = st.columns(2)

		with column1:

			if 'num_cons' not in st.session_state:
				num_cons = st.number_input('How many constraints do you have?',min_value=0, max_value=20,step=1)
			else:
				num_cons = st.number_input('How many constraints do you have?',min_value=0, max_value=20,step=1, value=st.session_state['num_cons'])
			
			# if 	num_cons == 0:
				# st.session_state["constraints"]={}
			valide = st.button('Submit')

			#If the user already used the app and has saved the constraints
		with column2:
			constraint_file = st.file_uploader("If you already saved constraints", type = ['.csv'])
			if constraint_file:
				constraint_csv = pd.read_csv(constraint_file)
				st.session_state['num_cons'] = len(constraint_csv)
				st.session_state['text_cons'] = constraint_csv.iloc[:,1]
				st.session_state['input_const'] = constraint_csv.iloc[:,2]
				st.session_state["check2"] = True
				st.session_state['test'] = True


		if valide:
			st.session_state['test'] = True
			st.session_state['num_cons'] = num_cons

		col1, col2 = st.columns(2)

		if 'num_cons' in st.session_state and "check2" not in st.session_state:
			text_cons=[]
			input_const=[]
			constraints= []
			constraint = []
			with st.form(key='constraint_selection'):
				with col1:
					for i in range(int(st.session_state['num_cons'])):
						text_cons.append(st.text_input('Constraint %s'%(i+1) +' : Write your constraint here',placeholder= 'x[:,i] + x[:,j]'))
					st.session_state['text_cons'] = text_cons
				with col2:
					for i in range(int(st.session_state['num_cons'])):
						input_const.append((st.text_input('Constraint %s'%(i+1) +' : Write your result here',placeholder= 'Ex: 1 or [0.2,1.5]')))
					st.session_state['input_const'] = input_const
				validate_cons = st.form_submit_button(label='Validate')
		elif 'text_cons' in st.session_state:
			text_cons=[]
			input_const=[]
			constraints= []
			constraint = []
			with st.form(key='constraint_selection'):
				with col1:
					for i in range(int(st.session_state['num_cons'])):
						if i > len(st.session_state["text_cons"])-1:
							text_cons.append(st.text_input('Constraint %s'%(i+1) +' : Write your constraint here',placeholder= 'x[:,i] + x[:,j]'))
						else:
							text_cons.append(st.text_input('Constraint %s'%(i+1) +' : Write your constraint here',value = st.session_state['text_cons'][i],placeholder= 'x[:,i] + x[:,j]'))							

				with col2:
					for i in range(int(st.session_state['num_cons'])):
						if i>len(st.session_state["input_const"])-1:
							input_const.append((st.text_input('Constraint %s'%(i+1) +' : Write your result here',placeholder= 'Ex: 1 or [0.2,1.5]')))
						else:
							input_const.append((st.text_input('Constraint %s'%(i+1) +' : Write your result here',value = st.session_state['input_const'][i],placeholder= 'Ex: 1 or [0.2,1.5]')))
				
				if 'text_cons' in st.session_state and st.session_state['num_cons']<len(st.session_state["text_cons"]):
					st.warning("The number of constraints saved are not the same than the constraints actually shown, you must validate your new selection.")
				validate_cons = st.form_submit_button(label='Validate')
		if st.session_state['test'] and validate_cons:
			st.session_state["check2"] = True
			st.session_state['text_cons'] = text_cons
			st.session_state['input_const'] = input_const
			for i in range(len(text_cons)):
				if "[" in input_const[i]:
					born_min = '-(%s'%text_cons[i] + ')+ %s'%input_const[i].split(",")[0].split("[")[1]
					constraint.append(born_min)
					born_max = '%s'%text_cons[i] + '- %s'%input_const[i].split(",")[1].split("]")[0]
					constraint.append(born_max)
				else:
					born_min = '-(%s'%text_cons[i] + ')+ %s'%input_const[i]
					constraint.append(born_min)
					born_max = '%s'%text_cons[i] + '- %s'%input_const[i]
					constraint.append(born_max)

			for i in range(len(constraint)):
				constraints.append({'name': 'constr_%s'%(i+1), 'constraint': '%s'%constraint[i]})
			st.session_state["constraints"] =constraints

		if "check2" in st.session_state:
			combined = {'Constraint': text_cons, 'Result':input_const}
			combined = pd.DataFrame(data=combined)

			data_const_csv = convert_feat_lim(combined)		

			st.download_button(
				label = 'Save your constraint for next time',
				data = data_const_csv,
				file_name = "constraint.csv",
				mime = 'text/csv')

		# Bayesian Optimisation

		st.subheader('Optimisation')

		# Ask the user how many target does he want to optimise
		if 'num_target' not in st.session_state:
			num_target = st.number_input('How many target do you want to optimise? (max 3)',min_value=1, max_value=len(st.session_state['target_selected']),step=1)
			st.session_state['num_target'] = int(num_target)
		else:
			num_target = st.number_input('How many target do you want to optimise? (max 3)',min_value=1, max_value=len(st.session_state['target_selected']),step=1, value = int(st.session_state['num_target']))
			st.session_state['num_target'] = num_target
		
		# Open as many selectbox as the user choose above
		target={}
		opti={}
		Y_init={}
		min_max = np.array(['minimize','maximize'])
		target = np.array([target_selected]).reshape(-1,1)
	
		if "Check target" not in st.session_state:
			for i in range(int(num_target)):
				target[i] = st.selectbox('Select which target you want to optimise, be careful to not choose the same target twice', target_selected,key=(i+1))
				st.session_state['Target_%s'%i]=np.where(target==target[i])[0][0]
				opti[i] = st.selectbox('Select if you want to maximize/minimize this target ', min_max, key=(i+5))
				st.session_state['Opti_%s'%i]=np.where(min_max==opti[i])[0][0]
		else:
			for i in range(int(num_target)):
				target[i] = st.selectbox('Select which target you want to optimise, be careful to not choose the same target twice', target_selected,key=(i+1), index=int(st.session_state['Target_%s'%i]))
				st.session_state['Target_%s'%i]=np.where(target==target[i])[0][0]
				opti[i] = st.selectbox('Select if you want to maximize/minimize this target ', min_max, key=(i+5), index=int(st.session_state['Opti_%s'%i]))
				st.session_state['Opti_%s'%i]=int(np.where(min_max==opti[i])[0][0])		

		# If the user chose only 1 target
		if num_target==1 and opti[0]=='maximize':
			Y_init = 1/(data_file_target.loc[:,target[0]].values.reshape(-1,  1)+ np.finfo(float).eps) # + machine error to avoid the possible division by 0
		elif num_target==1 and opti[0]== 'minimize':
			Y_init = data_file_target.loc[:,target[0]].values.reshape(-1,  1)
		else: # if the user chose 2 or 3
			for i in range(int(num_target)): 
				if opti[i]=='maximize':
					Y_init[i] = 1/(data_file_target.loc[:,target[i]].values.reshape(-1,  1) + np.finfo(float).eps) # + machine error to avoid the possible division by 0
				else:
					Y_init[i] = data_file_target.loc[:,target[i]].values.reshape(-1,  1)
			#This part is the sliders to let the user choose the factor between the different targets
			if num_target==2:
				a = st.slider('Select the importance factor of %s'%target[0][0]+' in pourcentage, compare to %s'%target[1][0], 0,100,50,1)
				b=100-a
				if (a+b)!=100:
					st.warning('The sum of the two slider must be equal to 100')
				Y_init=(Y_init[0]**(a/100)*(Y_init[1]**(b/100)))
			else:
				a = st.slider('Select the importance factor of %s'%target[0][0]+' in %', 0,100,34,1)
				b = st.slider('Select the importance factor of %s'%target[1][0]+' in %', 0,(100-a),33,1)
				c = st.slider('Select the importance factor of %s'%target[2][0]+' in %', 0,(100-a-b),33,1)

				if (a+b+c)!=100:
					st.warning('The sum of the two slider must be equal to 100')
				Y_init=(Y_init[0]**(a/100)*(Y_init[1]**(b/100))*(Y_init[2]**(c/100)))

		# Allow the user to choose how many optimise sample does he want
		if 'iter_count' not in st.session_state:
			iter_count = st.slider('Select the number of sample that you want', 1, 20, 10, 1)
			st.session_state['iter_count'] = iter_count
		else:
			iter_count = st.slider('Select the number of sample that you want', 1, 20, int(st.session_state['iter_count']), 1)
			st.session_state['iter_count'] = iter_count
				

		###################
		#### Code for Bayesian Optimisation
		###################

		execute = st.button('Execute the Bayesian Optimisation')
		X_init = data_file_feature.loc[:,:].values

		if execute:
			st.session_state["Check target"] = True
			pending_X = np.empty(shape = (0, len(st.session_state['feature_lim'].iloc[:,0])))
			Next_point = np.empty(shape = (0, 1))
			predict_values = np.empty(shape = (0, 1))
			std_values = np.empty(shape = (0, 1))

			for iitc in range(iter_count):
				bo_step = gpopt.methods.BayesianOptimization(f = None, 
			                                                 domain = space,
			                                                 constraints = st.session_state["constraints"],
			                                                 X = X_init, 
			                                                 Y = Y_init, 
			                                                 acquisition_type = 'EI',
			                                                 verbosity=True,
			                                                 verbosity_model=True,
			                                                 normalize_Y = False,
			                                                 exact_feval = True,
			                                                 de_duplication = True) # Enable treatment of pending and ignored locations
				x_next = bo_step.suggest_next_locations(pending_X = pending_X)
				pending_X = np.vstack((pending_X, x_next))
				post_mean, std = bo_step.model.predict(x_next) #bo_step.model.predict() return the posterior value and the standard deviation but bo_step.model.model.predict() return the same posterior but with the variance (std**2)
				predict_values = np.vstack((predict_values,post_mean))
				std_values = np.vstack((std_values,std))

			proposed_list_next_opti = pd.DataFrame(pending_X, columns = data_file_feature.columns)
			st.write('The result of the bayesian optimisation are :')
			st.session_state['proposed_list']= proposed_list_next_opti

		if 'proposed_list' in st.session_state: 
			# Display the next sample find with the Bayesian Optimisation
			
			st.dataframe(st.session_state['proposed_list'])
			proposed_list_csv = convert_feat_lim(st.session_state['proposed_list'])

			# Downloading the result of the Bayesian Optimisation

			st.write('Click below to download the CSV with the proposed data for optimisation.')
			st.download_button(
				label = 'Click here',
				data = proposed_list_csv,
				file_name = 'proposed_list.csv',
				mime = 'text/csv')

########## Bayesian with prediction model

		st.subheader('Bayesian Optimisation using the prediction')

		st.write("In this part, the bayesian optimisation will initialise not with your dataset but using values from the prediction model obtained in the other page. The purpose of this operation is to unbiase the dataset. It is recommanded to use this part only if you obtained a good prediction model in the second page, otherwise do not use this part.")
		st.write("Now let's try to use the prediction you did in the previous page with the bayesian optimisation.\n\n"
			"Choose the parameters that have showm better performance in the Prediction Page.")
		st.write("It works only if you want to optimise one target") 

		methods = ['ElasticNet', 'RandomForestRegressor', 'XGBRegressor']

		X_init_2 = X_init
		Y_init_2 = Y_init

		with st.form('Prediction'):

			target = st.selectbox('Select which target you want to predict',target_selected)

			method = st.selectbox('Select which method of prediction you want to use',methods)

			crossval = st.selectbox('Select which method of cross validation you want to use',['LeaveOneOut','K-Fold'])
			
			k_num = st.number_input('Choose how many subsets do you want to use, it has an impact only if you selected K-fold',min_value=2,max_value=len(st.session_state['data_file_selected'].iloc[:,0])-1,value=3)

			predict = st.form_submit_button('Submit your selection')	

		if method == 'ElasticNet':
			regressor = ElasticNet(random_state = 0)
		elif method == 'RandomForestRegressor':
			regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
		elif method == 'XGBRegressor':
			regressor = XGBRegressor(n_estimators = 100, seed = 0)

		if crossval=='LeaveOneOut':
			crossvalidation = LeaveOneOut()
		elif crossval=='K-Fold':
			crossvalidation = KFold(n_splits=int(k_num), shuffle=True, random_state=0)

		clf = make_pipeline(StandardScaler(), regressor)

		mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
		mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
		scorer = {'mae': mae_scorer, 'mse': mse_scorer}

		def cross_val_est_fn(clf, x, y, cv):
		    predictions = cross_val_predict(estimator = clf, 
		                                    X = x, 
		                                    y = y, 
		                                    cv = cv)
		    validation_arrays = cross_validate(estimator = clf, 
		                                      X = x, 
		                                      y = y, 
		                                      cv = cv, 
		                                      scoring = scorer, 
		                                      return_estimator =True)   
		    
		    test_mae, test_mse, estimator = validation_arrays['test_mae'], validation_arrays['test_mse'], validation_arrays['estimator']

		    return predictions, -test_mae, np.sqrt(-test_mse), estimator

		if predict:
			pred_, test_mae, test_rmse, est_S = cross_val_est_fn(clf = clf,x=X_init,y=Y_init,cv=crossvalidation)
			st.session_state['est_S'] = est_S

		def f_mod(X):
			est_S = st.session_state['est_S']
			Popt_predict_list = []
			for ir in range(len(est_S)):
				pred_S_tmp = est_S[ir].predict(X)
				Popt_predict_list.append(pred_S_tmp)

			Popt_predict_mean = np.mean(Popt_predict_list)

			return Popt_predict_mean

		execute2 = st.button('Execute the Bayesian Optimisation with Prediction')

		# Code for Bayesian Optimisation

		if execute2:
			
			n_exps = 10 # nombre n de samples pour initialiser le Gaussian Process
			st.session_state['target[i]']='ok'
			pending_X_2 = np.empty(shape = (0, len(st.session_state['feature_lim'].iloc[:,0])))
			Next_point_2 = np.empty(shape = (0, 1)) # A supprimer apres
			X_init = data_file_feature.loc[:,:].values	
			
			def function():
				bo_step_2 = gpopt.methods.BayesianOptimization( f =f_mod,
				                                             domain = space,
				                                             constraints = st.session_state["constraints"], 
				                                             X = X_init_2, 
				                                             Y = Y_init_2, 
				                                             acquisition_type = 'EI',
				                                             normalize_Y = False,
				                                             exact_feval = True,
				                                             initial_design_numdata = n_exps*10,
				                                             de_duplication = True)
				
				file_name = 'evaluation_file.csv'
				run_opti = bo_step_2.run_optimization(max_iter=iter_count,verbosity=True, evaluations_file = file_name)
				evaluations_file_csv = pd.read_csv(file_name,sep='	')
				next_samples = pd.DataFrame(evaluations_file_csv[-iter_count:])
				os.remove('evaluation_file.csv')
				return next_samples
			next_samples = function()
			
			st.write('The result of the bayesian optimisation using the prediction are :')
			proposed_list_next_opti_with_predi = pd.DataFrame(np.array(next_samples.iloc[:,2:]), columns = data_file_feature.columns)
			st.session_state['proposed_list_next_opti_with_predi']= proposed_list_next_opti_with_predi

		if 'proposed_list_next_opti_with_predi' in st.session_state: # Display the next sample find with the Bayesian Optimisation

			st.dataframe(st.session_state['proposed_list_next_opti_with_predi'])

			proposed_list_with_predi_csv = convert_feat_lim(st.session_state['proposed_list_next_opti_with_predi'])

			# Downloading the result of the Bayesian Optimisation

			st.write('Click below to download the CSV with the proposed data for optimisation.')
			st.download_button(
				label = 'Click here',
				data = proposed_list_with_predi_csv,
				file_name = 'proposed_list_with_predi.csv',
				mime = 'text/csv')

if choice == 'About':
	
	
	def read_markdown_file(markdown_file):
	    return Path(markdown_file).read_text()

	intro_markdown = read_markdown_file("../ABOUT.md")
	st.markdown(intro_markdown, unsafe_allow_html=True)

	
