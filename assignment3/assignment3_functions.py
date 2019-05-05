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
from sklearn import (svm, ensemble, tree,
                     linear_model, neighbors, naive_bayes, dummy)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import ParameterGrid


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
    return df.isna().sum(axis=0)

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


def cols_to_dummy(df, col_list, val):

    for col in col_list:
        df[col] = df[col].apply(lambda x: 1 if x == val else 0)

    return df


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


def temporal_validate(start_time, end_time, prediction_windows):
    '''
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    temp_split = []

    for prediction_window in prediction_windows:
        windows = 1
        test_end_time = start_time_date
        while (end_time_date >= test_end_time + relativedelta(months=+prediction_window)):
            train_start_time = start_time_date
            train_end_time = train_start_time + windows * relativedelta(months=+prediction_window) - relativedelta(days=+1)
            test_start_time = train_end_time + relativedelta(days=+1)
            test_end_time = test_start_time  + relativedelta(months=+prediction_window) - relativedelta(days=+1)
            temp_split.append([train_start_time,train_end_time,test_start_time,test_end_time,prediction_window])


            windows += 1

    return temp_split


def temporal_split(df, time_var, selected_y, train_start, train_end, test_start, test_end):
    '''
    '''
    train_data = df[(df[time_var] >= train_start) & (df[time_var] <= train_end)]
    train_data.drop([time_var], axis=1)
    y_train = train_data[selected_y]
    x_train = train_data.drop([selected_y, time_var], axis=1)
    test_data = df[(df[time_var] >= test_start) & (df[time_var] <= test_end)]
    test_data.drop([time_var], axis=1)
    y_test = test_data[selected_y]
    x_test = test_data.drop([selected_y, time_var], axis=1)

    return x_train, x_test, y_train, y_test


def process(df, val):
    '''
    '''
    processed_df = df.copy(deep=True)
    na_cols = [col for col in processed_df if col.startswith('val')]
    for col in na_cols:
        df[col] = df[col].fillna(df[col].median())

    return df

def build_classifier(x_train, y_train, models_to_run, params_dict=None):

    models_dict = {'RF': ensemble.RandomForestClassifier,
                   'LR': linear_model.LogisticRegression,
                   'KNN': neighbors.KNeighborsClassifier,
                   'DT': tree.DecisionTreeClassifier,
                   'SVM': svm.LinearSVC,
                   'AB': ensemble.AdaBoostClassifier,
                   }

    fitted_models = {}

    for model in models_to_run:
        print('running model,')
        print(model)
        if params_dict:
            params = params_dict[model]
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

    return precision_at_k, accuracy_at_k, recall_at_k


def f1_at_k(y_test, y_pred_scores, k):
    '''
    '''
    y_pred_at_k = generate_binary_at_k(y_pred_scores, k)
    f1_at_k = metrics.f1_score(y_test, y_pred_scores)

    return f1_at_k


def joint_sort_descending(l1, l2):
    '''
    Sorts y_test and y_pred in descending order of probability.
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def create_eval_table(x_test, y_test, fitted_models, k_list, models_of_interest):
    '''
    '''
    eval_dict = {'model': [], 'k': [], 'precision': [], 'accuracy': [], 'recall': [], 'auc': []}

    for model in models_of_interest:
        print('running through...')
        print(model)
        if model == "SVM":
            y_pred_scores = fitted_models[model].decision_function(x_test)
        else:
            y_pred_scores = fitted_models[model].predict_proba(x_test)[:,1]

        for k in k_list:
            eval_dict['k'].append(k)
            precision_at_k, accuracy_at_k, recall_at_k = evaluation_scores_at_k(y_test, y_pred_scores, k)
            # f1 = f1_at_k(y_test, y_pred_scores, k)
            eval_dict['model'].append(model)
            eval_dict['precision'].append(precision_at_k)
            eval_dict['accuracy'].append(accuracy_at_k)
            eval_dict['recall'].append(recall_at_k)
            # eval_dict['f1'].append(f1)
            eval_dict['auc'].append(metrics.roc_auc_score(y_test, y_pred_scores))

    return pd.DataFrame.from_dict(eval_dict)


def plot_precision_recall_n(y_test, y_pred_scores, model_name):
    '''
    Plots precision-recall curve for a given model.
    '''
    y_score = y_pred_scores
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


 
def create_temporal_eval_table(models_to_run, classifiers, parameters, df, selected_y, temp_split, time_var, outfile):
    '''
    '''
    results_df = pd.DataFrame(columns=('train_start', 'train_end', 'test_start', 'test_end', 'model_type', 'classifier', 'train_size', 'test_size', 'auc-roc',
        'p_at_1', 'a_at_1', 'r_at_1', 'p_at_2', 'a_at_2', 'r_at_2', 'p_at_5', 'a_at_5', 'r_at_5', 'p_at_10', 'a_at_10', 'r_at_10', 
        'p_at_20', 'a_at_20', 'r_at_20', 'p_at_30', 'a_at_30', 'r_at_30', 'p_at_50', 'a_at_50', 'r_at_50'))

    params = []

    for timeframe in temp_split:
        train_start, train_end, test_start, test_end = timeframe[0], timeframe[1], timeframe[2], timeframe[3]
        x_train, x_test, y_train, y_test = temporal_split(df, time_var, selected_y, train_start, train_end, test_start, test_end)
        x_train = process(x_train, 'students_reached')
        x_test = process(x_test, 'students_reached')
        for index, classifier in enumerate([classifiers[x] for x in models_to_run]):
                print("Running through model {}...".format(models_to_run[index]))
                parameter_values = parameters[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    params.append(p)
                    try:
                        classifier.set_params(**p)
                        if models_to_run[index] == 'SVM':
                            y_pred_probs = classifier.fit(x_train, y_train).decision_function(x_test)
                        else:
                            y_pred_probs = classifier.fit(x_train, y_train).predict_proba(x_test)[:,1]
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                        precision_1, accuracy_1, recall_1 = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, 1.0)
                        precision_2, accuracy_2, recall_2 = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, 2.0)
                        precision_5, accuracy_5, recall_5 = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, 5.0)
                        precision_10, accuracy_10, recall_10 = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, 10.0)
                        precision_20, accuracy_20, recall_20 = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, 20.0)
                        precision_30, accuracy_30, recall_30 = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, 30.0)
                        precision_50, accuracy_50, recall_50 = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, 50.0)
                        results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end,
                                                           models_to_run[index],
                                                           classifier,
                                                           y_train.shape[0], y_test.shape[0],
                                                           metrics.roc_auc_score(y_test_sorted, y_pred_probs),
                                                           precision_1, accuracy_1, recall_1,
                                                           precision_2, accuracy_2, recall_2,
                                                           precision_5, accuracy_5, recall_5,
                                                           precision_10, accuracy_10, recall_10,
                                                           precision_20, accuracy_20, recall_20,
                                                           precision_30, accuracy_30, recall_30,
                                                           precision_50, accuracy_50, recall_50]

                    except IndexError as e:
                        print('Error:',e)
                        continue

        results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end, "baseline", '', '', '',
                        y_test.sum()/len(y_test), '', '', '', '', '', '', '', '', '', '', '', '', '','', '', '', '',
                        '', '', '', '']

    
    results_df.to_csv(outfile)

    return results_df, params
















