'''
CAPP 30254 Building the Machine Learning Pipeline

Bhargavi Ganesh
'''
import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

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

def explore_data(df):
	'''
	This function describes the data, summarizes NA values,
	and finds correlations between variables
	'''
	summary_stats = df.describe()
	na_summary = df.isna().any()
	correlations = df.corr()
	# if df[var_to_describe].name == 'int64':
		# numeric_var_range = summary_stats[var_to_describe]['max'] - summary_stats[var_to_describe]['min']
		# var_hist = plt.hist(df[var_to_describe], color = 'blue', edgecolor = 'black', bins = (numeric_var_range/hist_increments).astype('int64'))

	return summary_stats, na_summary, correlations

def data_no_ol(df):
	'''
	This function takes a dataframe, and returns 
	summary table without outliers
	'''
	Q1 = df.quantile(0.25)
	Q3 = df.quantile(0.75)
	IQR = Q3 - Q1
	df_out = df.copy(deep=True)
	df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

	return df_out

def summary_figures(df, var_to_describe):
	'''
	This function takes a dataframe, creates histograms 
	to understand the distribution of numeric variables,
	and boxplots to find outliers
	'''
	var_hist = plt.hist(df[var_to_describe], color = 'blue', edgecolor = 'black')
	var_boxplot = sns.boxplot(x=df[var_to_describe])

	return var_hist, var_boxplot

def pre_process(df):
	'''
	This function fills in missing values in the data
	'''
	processed_df = df.copy(deep=True)
	processed_df = processed_df.fillna(processed_df.median())

	return processed_df

def discretize(df, vars_to_discretize):
	'''
	This function discretizes a continous variable
	'''
	# age_bins = [0, 21, 41, 51, 62, 109]
	for item in vars_to_discretize:
		new_label = item + '_category'
		df[new_label] = pd.qcut(df[item], 5)

	return df

def categorize(df, vars_to_categorize):
	'''
	This function takes a list of categorical variables and creates a 
	binary/dummy variable from it
	'''
	df_with_categorical = pd.get_dummies(df, columns=vars_to_categorize)

	return df_with_categorical

	









