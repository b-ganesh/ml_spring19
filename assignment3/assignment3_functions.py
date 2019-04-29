'''
CAPP 30254 Building the Machine Learning Pipeline

Bhargavi Ganesh
'''
import os 
import pandas as pd
import numpy as np 
import math
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve


#plots code adapted from: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

def file_to_dataframe(filename):
	'''
	Takes a filename and returns a pandas dataframe.
	
	Input:
		filename

	Returns:
		pandas dataframe

	'''

	if os.path.exists(filename):
		return pd.read_csv(filename)

def na_summary(df):
	''''
	Takes a dataframe and returns a table
	showing which columns have NAs.

	Input: 
		pandas dataframe

	Returns:
		table with nas
	'''
	return df.isna().any()

def describe_data(df, vars_to_describe=None):
	'''
	This function describes the data, providing
	basic descriptive statistics such as min,
	max, median, mean, etc.

	Input:
		pandas dataframe
		(optional) list of variables to describe

	Returns:
		table with min, max, mean, median, etc
		for each column in the specified df
	'''
	if vars_to_describe:
		df = df[vars_to_describe]

	return df.describe()

def histograms(df, vars_to_describe=None):
	'''
	Function that plots histogram of every variable in df.

	Input:
		pandas dataframe
		(optional) list of variables to describe
	'''
	if vars_to_describe:
		df = df[vars_to_describe]

	plt.rcParams['figure.figsize'] = 16, 12
	df.hist()
	plt.show()

def correlations(df, vars_to_describe=None):
	'''
	This function takes a dataframe and returns
	a correlation matrix with the specified variables.

	Input:
		pandas df
		(optional) list of variables to describe
	'''
	if vars_to_describe:
		df = df[vars_to_describe]

	return df.corr()

def correlation_matrix(correlations):
	'''
	This function takes a correlation table
	and plots a correlation matrix.

	Input:
		correlations: correlation table
	'''
	plt.rcParams['figure.figsize'] = 10, 10
	names = correlations.columns
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0,len(names),1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names, rotation=30, rotation_mode='anchor', ha='left')
	ax.set_yticklabels(names)
	plt.show()

def pairplot(df, vars_to_describe):
	'''
	This function takes a dataframe and variables
	to describe and plots a pairplot showing the 
	relationship between variables.

	Inputs:
		pandas dataframe
		(optional) list of variables to describe
	'''
	plt.rcParams['figure.figsize']=(20,10)
	sns.pairplot(df, vars=vars_to_describe, dropna=True, height=3.5)
	plt.show()	

def boxplots(df, vars_to_describe=None):
	'''
	This function takes a dataframe and variables
	to describe and plots boxplots for all the columns
	in the df.

	Inputs:
		pandas dataframe
		(optional) list of variables to describe
	'''
	if vars_to_describe:
		df = df[vars_to_describe]

	plt.rcParams['figure.figsize'] = 16, 12
	df.plot(kind='box', subplots=True, 
	layout=(5, math.ceil(len(df.columns)/5)), 
	sharex=False, sharey=False)
	plt.show()

def identify_ol(df, vars_to_describe=None):
	'''
	This function takes a dataframe, and returns a table of outliers

	Inputs:
		pandas dataframe
		(optional) list of variables to describe

	Returns:
		pandas dataframe with outliers
	'''
	subset_df = df.copy(deep=True)
	if vars_to_describe:
		subset_df = subset_df[vars_to_describe]
	Q1 = subset_df.quantile(0.25)
	Q3 = subset_df.quantile(0.75)
	IQR = Q3 - Q1
	df_out = \
	subset_df[((subset_df < (Q1 - 1.5 * IQR)) | \
	(subset_df > (Q3 + 1.5 * IQR))).any(axis=1)]

	return df_out

def pre_process(df):
	'''
	This function takes a dataframe and fills in missing values
	for each column in the data, based on the median of the column.

	Inputs:
		pandas dataframe

	Returns:
		pandas dataframe with missing values filled
	'''
	processed_df = df.copy(deep=True)
	processed_df = processed_df.fillna(processed_df.median())

	return processed_df

def discretize(df, vars_to_discretize, num_bins=10):
	'''
	This function takes a dataframe and a list of variables
	to discretize and discretizes each continous variable.

	Inputs:
		pandas dataframe
		list of variables to discretize
		(optional) number of bins

	Returns:
		pandas dataframe with discretized variables
	'''
	for item in vars_to_discretize:
		new_label = item + '_discrete'
		df[new_label] = pd.qcut(df[item], num_bins)

	return df

def categorize(df, vars_to_categorize):
	'''
	This function takes a dataframe and a list of categorical variables 
	and creates a binary/dummy variable from it

	Inputs:
		pandas dataframe
		list of variables to categorize

	Returns:
		pandas dataframe with dummy variables
	'''
	df_with_categorical = pd.get_dummies(df, columns=vars_to_categorize)

	return df_with_categorical

def split_data(df, selected_features, selected_y, test_size):
	'''
	This function takes a dataframe, a list of selected features, 
	a selected y variable, and a test size, and returns a 
	training set and a testing set of the data.

	Inputs:
		pandas dataframe
		list of selected x variables
		selected y variable

	Returns:
		x-variable training dataset, y-variable training dataset,
		x-variable testing dataset,  y-variable testing dataset
	'''
	x = df[selected_features]
	y = df[selected_y]
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=test_size)

	return x_train, x_test, y_train, y_test


# def temporal_split():

def build_classifier(x_train, x_test, y_train, y_test, models_to_run, params=None):

	models_dict = {'RandomForest': ensemble.RandomForestClassifier,
				   'LogisticRegression': linearmodel.LogisticRegression,
				   'KNeighborsClassifier': neighbors.KNeighborsClassifier,
				   'DecisionTreeClassifier': tree.DecisionTreeClassifier,
				   'SVM': svm.LinearSVC,
				   'Boosting': ensemble.AdaBoostClassifier,
				   'Bagging': ensemble.BaggingClassifier
				   }

	fitted_models = {}

	for model in models_to_run:
		if params:
			model_obj = models_dict[model](**params) 
		else:
			model_obj = models_dict[model]()

		trained = model_obj.fit(x_train, y_train)

		fitted_models[model] = trained

	return fitted_models


def generate_binary_at_k(y_pred_scores, k):
    '''
    Converts probability score into binary outcome measure based upon cutoff.
    '''
    cutoff_index = int(len(y_pred_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_pred_scores))]

    return test_predictions_binary

def evaluation_scores_at_k(y_test, y_pred_scores, k):
	'''
	'''
	y_pred_at_k = generate_binary_at_k(y_pred_scores, k)
	precision_at_k = metrics.precision_score(y_test, y_pred_at_k)
	accuracy_at_k = metrics.accuracy_score(y_test, y_pred_at_k)
	recall_at_k = metrics.recall_score(y_test, y_pred_at_k)
	f1_at_k = metrics.f1_score(y_test, y_pred_at_k)
	roc_auc = metrics.roc_auc_curve(y_test, y_pred_at_k)

	return precision_at_k, accuracy_at_k, recall_at_k, f1_at_k


def create_eval_table(x_train, x_test, y_train, y_test, fitted_models, k):
	'''
	'''
	eval_dict = {'model': [], 'precision': [], 'accuracy': [], 'recall': [], 'f1': [], 'auc_score': []}

	for model in fitted_models:
		y_pred_scores = fitted_models.predict_proba(x_test)[:,1]
		precision_at_k, accuracy_at_k, recall_at_k, f1_at_k, roc_auc = evaluation_scores_at_k(y_test, y_pred_scores, k)
		eval_dict['model'].append(model)
		eval_dict['precision'].append(precision_at_k)
		eval_dict['accuracy'].append(accuracy_at_k)
		eval_dict['recall'].append(recall_at_k)
		eval_dict['f1'].append(f1_at_k)
		eval_dict['auc_score'].append(roc_auc)

	return pd.dataframe_from_dict(eval_dict)

# def create_temporal_eval_table(x_train, x_test, y_train, y_test, fitted_models):


def plot_precision_recall_n(y_test, y_pred_scores, model_name):
    '''
    Plots precision-recall curve for a given model.
    '''
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_scores)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_pred_scores[y_pred_scores>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

 






