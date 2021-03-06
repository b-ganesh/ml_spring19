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
		x, y, test_size=test_size, random_state=1)

	return x_train, x_test, y_train, y_test

def build_classifier(x_train, x_test, y_train, y_test, max_depth):
	'''
	This function builds a classifier using the decision trees module

	Inputs:
		x-variable training dataset, y-variable training dataset,
		x-variable testing dataset,  y-variable testing dataset
		max_depth: maximum depth of the tree

	Returns:
		tuple of predicted y values and decision tree object
	'''
	#Create Decision Tree classifier object
	dec_tree = DecisionTreeClassifier(
		criterion='entropy', max_depth=max_depth)

	#Train Decision Tree classifier
	dec_tree = dec_tree.fit(x_train, y_train)

	#Predict response for test dataset
	y_pred = dec_tree.predict(x_test)

	return y_pred, dec_tree

def evaluate_classifier(y_test, y_pred):
	'''
	This function takes the predicted y values and actual y values from 
	the test set and calculates the accuracy of the model

	Returns:
		accuracy score (percentage)
	'''
	return metrics.accuracy_score(y_test, y_pred)

 






