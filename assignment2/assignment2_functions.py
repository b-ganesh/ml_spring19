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

def explore_data(df, var_to_describe, hist_increments):
	'''
	This function describes the data, creates histogram and 
	density plots, and finds correlations between variables
	'''
	summary_stats = df.describe()
	na_summary = df.isna().any()
	correlations = df.corr()
	if df[var_to_describe].name == 'int64':
		numeric_var_range = summary_stats[var_to_describe]['max'] - summary_stats[var_to_describe]['min']
		var_hist = plt.hist(df[var_to_describe], color = 'blue', edgecolor = 'black', bins = (numeric_var_range/hist_increments).astype('int64'))


	return summary_stats, na_summary, var_hist, correlations


def summary_tables(df):


def pre_process(df):
	

		# plt.hist(df[var_to_describe], color = 'blue', edgecolor = 'black', bins = numeric_var_range/10)






