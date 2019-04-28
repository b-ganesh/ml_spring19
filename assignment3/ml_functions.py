'''
Assignment 3 functions
'''
import logging
import pdb
import numpy as np
from sklearn import (svm, ensemble, tree,
                     linear_model, neighbors, naive_bayes)
from sklearn.feature_selection import SelectKBest


TEST_GRID = {
            'RandomForest': {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'LogisticRegression': {'penalty': ['l1'], 'C': [0.01]},
            'SGDClassifier': { 'loss': ['perceptron'], 'penalty': ['l2']},
            'ExtraTrees': {'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'AdaBoost': {'algorithm': ['SAMME'], 'n_estimators': [1]},
            'GradientBoostingClassifier': {'n_estimators': [1], 'learning_rate' : [0.1], 'subsample' : [0.5], 'max_depth': [1]},
            'GaussianNB': {},
            'DecisionTreeClassifier': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'SVM': {'C':[0.01],'kernel':['linear']},
            'KNeighborsClassifier': {'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
            }

MODELS_TO_RUN = ['RandomForest', 'LogisticRegression', 'DecisionTreeClassifier']

def split_data(df, selected_features, selected_y, test_size, random_state):
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
        x, y, test_size=test_size, random_state)

    return x_train, x_test, y_train, y_test



def define_model(model, parameters, n_cores):
    if model == "RandomForest":
        return ensemble.RandomForestClassifier(
            n_estimators=parameters['n_estimators'],
            max_features=parameters['max_features'],
            criterion=parameters['criterion'],
            max_depth=parameters['max_depth'],
            min_samples_split=parameters['min_samples_split'],
            random_state=parameters['random_state'],
            n_jobs=n_cores)

    elif model == "RandomForestBagging":
        #TODO Make Model Bagging
        return ensemble.BaggingClassifier(
                    ensemble.RandomForestClassifier(
                        n_estimators=parameters['n_estimators'],
                        max_features=parameters['max_features'],
                        criterion=parameters['criterion'],
                        max_depth=parameters['max_depth'],
                        min_samples_split=parameters['min_samples_split'],
                        random_state=parameters['random_state'],
                        n_jobs=n_cores),
                    #Bagging parameters
                    n_estimators=parameters['n_estimators_bag'],
                    max_samples=parameters['max_samples'],
                    max_features=parameters['max_features_bag'],
                    bootstrap=parameters['bootstrap'],
                    bootstrap_features=parameters['bootstrap_features'],
                    n_jobs=n_cores
                    )

    elif model == "RandomForestBoosting":
        #TODO Make Model Boosting
        return ensemble.AdaBoostClassifier(
            ensemble.RandomForestClassifier(
                n_estimators=parameters['n_estimators'],
                max_features=parameters['max_features'],
                criterion=parameters['criterion'],
                max_depth=parameters['max_depth'],
                min_samples_split=parameters['min_samples_split'],
                random_state=parameters['random_state'],
                n_jobs=n_cores),
            #Boosting parameters
            learning_rate=parameters['learning_rate'],
            algorithm=parameters['algorithm'],
            n_estimators=parameters['n_estimators_boost']
            )

    elif model == 'SVM':
        return svm.SVC(C=parameters['C_reg'],
                       kernel=parameters['kernel'],
                       probability=True,
                       random_state=parameters['random_state'])

    elif model == 'LogisticRegression':
        return linear_model.LogisticRegression(
            C=parameters['C_reg'],
            penalty=parameters['penalty'],
            random_state=parameters['random_state'])

    elif model == 'AdaBoost':
        return ensemble.AdaBoostClassifier(
            learning_rate=parameters['learning_rate'],
            algorithm=parameters['algorithm'],
            n_estimators=parameters['n_estimators'],
            random_state=parameters['random_state'])

    elif model == 'ExtraTrees':
        return ensemble.ExtraTreesClassifier(
            n_estimators=parameters['n_estimators'],
            max_features=parameters['max_features'],
            criterion=parameters['criterion'],
            max_depth=parameters['max_depth'],
            min_samples_split=parameters['min_samples_split'],
            random_state=parameters['random_state'],
            n_jobs=n_cores)

    elif model == 'GradientBoostingClassifier':
        return ensemble.GradientBoostingClassifier(
            n_estimators=parameters['n_estimators'],
            learning_rate=parameters['learning_rate'],
            subsample=parameters['subsample'],
            max_depth=parameters['max_depth'],
            random_state=parameters['random_state'])

    elif model == 'GaussianNB':
        return naive_bayes.GaussianNB()

    elif model == 'DecisionTreeClassifier':
        return tree.DecisionTreeClassifier(
            max_features=parameters['max_features'],
            criterion=parameters['criterion'],
            max_depth=parameters['max_depth'],
            min_samples_split=parameters['min_samples_split'],
            random_state=parameters['random_state'])

    elif model == 'SGDClassifier':
        return linear_model.SGDClassifier(
            loss=parameters['loss'],
            penalty=parameters['penalty'],
            random_state=parameters['random_state'],
            n_jobs=n_cores)

    elif model == 'KNeighborsClassifier':
        return neighbors.KNeighborsClassifier(
            n_neighbors=parameters['n_neighbors'],
            weights=parameters['weights'],
            algorithm=parameters['algorithm'],
            n_jobs=n_cores)

    else:
        raise ConfigError("Unsupported model {}".format(model))



def gen_model(train_x, train_y, test_x, model, parameters, n_cores=1):
    """Trains a model and generates risk scores.
    Args:
        train_x: training features
        train_y: training target variable
        test_x: testing features
        model (str): model type
        parameters: hyperparameters for model
        n_cores (Optional[int]): number of cores to us
    Returns:
        result_y: predictions on test set
        modelobj: trained model object
    """
    modelobj = define_model(model, parameters, n_cores)
    modelobj.fit(train_x, train_y)
    result_y_probability = modelobj.predict_proba(test_x)
    result_y_binary = modelobj.predict(test_x)

    return result_y_probability[:, 1], result_y_binary, modelobj

def run_models(df, models_list, selected_features, selected_y, test_size, random_state=1):
    for model in models_list:
        train_x, train_y, text_x, test_y = split_data(df, selected_features, selected_y, test_size)
        






#run loop of all the models. if temporal validate, include that as well. have functions for precision/recall/etc too.
