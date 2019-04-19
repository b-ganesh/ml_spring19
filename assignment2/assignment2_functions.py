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
	Takes a filename and returns a pandas dataframe
	
	Input:
		filename

	Returns:
		pandas dataframe

	'''

	if os.path.exists(filename):
		return pd.read_csv(filename)

def na_summary(df):
	return df.isna().any()

def describe_data(df, vars_to_describe=None):
	'''
	This function describes the data, summarizes NA values,
	and finds correlations between variables
	'''
	if vars_to_describe:
		df = df[vars_to_describe]

	return df.describe()

def histograms(df, vars_to_describe=None):
	'''
	Function that plots histogram of every variable in df
	'''
	if vars_to_describe:
		df = df[vars_to_describe]

	plt.rcParams['figure.figsize'] = 16, 12
	df.hist()
	plt.show()

def correlations(df, vars_to_describe=None):
	'''
	'''
	if vars_to_describe:
		df = df[vars_to_describe]

	return df.corr()

def correlation_matrix(df, correlations):
	'''
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
	Pairplot with variables of interest
	'''
	plt.rcParams['figure.figsize']=(20,10)
	sns.pairplot(df, vars=vars_to_describe, dropna=True, height=3.5)
	plt.show()	

def boxplots(df, vars_to_describe=None):
	'''
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
	'''
	subset_df = df.copy(deep=True)
	if vars_to_describe:
		subset_df = subset_df[vars_to_describe]
	# subset_df = df[df.columns[~df.columns.isin(['PersonID','SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'zipcode'])]]
	Q1 = subset_df.quantile(0.25)
	Q3 = subset_df.quantile(0.75)
	IQR = Q3 - Q1
	df_out = subset_df[((subset_df < (Q1 - 1.5 * IQR)) |(subset_df > (Q3 + 1.5 * IQR))).any(axis=1)]

	return df_out

def pre_process(df):
	'''
	This function fills in missing values in the data
	'''
	processed_df = df.copy(deep=True)
	processed_df = processed_df.fillna(processed_df.median())

	return processed_df

def discretize(df, vars_to_discretize, n_bins):
	'''
	This function discretizes a continous variable
	'''
	for item in vars_to_discretize:
		new_label = item + '_discrete'
		df[new_label] = pd.qcut(df[item], n_bins)

	return df

def categorize(df, vars_to_categorize):
	'''
	This function takes a list of categorical variables and creates a 
	binary/dummy variable from it
	'''
	df_with_categorical = pd.get_dummies(df, columns=vars_to_categorize)

	return df_with_categorical

def split_data(df, selected_features, selected_y, test_size):
	'''
	This function takes a dataframe, a list of selected features, 
	a selected y variable, and a test size, and returns a 
	training set and a testing set of the data.
	'''
	x = df[selected_features]
	y = selected_y
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_siz, random_state=1)

	return x_train, x_test, y_train, y_test

def build_classifier(x_train, x_test, y_train, y_test):
	'''
	This function builds a classifier using the decision trees module
	'''
	#Create Decision Tree classifier object
	dec_tree = DecisionTreeClassifier()

	#Train Decision Tree classifier
	dec_tree = dec_tree.fit(x_train, y_train)

	#Predict response for test dataset
	y_pred = dec_tree.predict(x_test)

	return y_pred

def evaluate_classifer(y_test, y_pred):
	'''
	This function takes the predicted y values and y values from 
	the test set and calculates the accuracy of the model
	'''
	return metrics.accuracy_score(y_test, y_pred)

# def visualize_tree(dec_tree, x_train):
# 	viz = tree.export_graphviz(
# 		dec_tree, feature_names=x_train.columns, class_names=class_names, rounded=True, filled=True)

# 	with open("tree.dot") as f:
# 		dot_graph = f.read()
# 		graph = graphviz.Source(dot_graph)

# 	return graph







