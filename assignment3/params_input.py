'''
Parameters for 
'''
TEST_GRID = {
            'RandomForest': {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'LogisticRegression': {'penalty': ['l1'], 'C': [0.01]},
            'Boosting': {'algorithm': ['SAMME'], 'n_estimators': [1]},
            'DecisionTreeClassifier': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'SVM': {'C':[0.01],'kernel':['linear']},
            'KNeighbors': {'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
            }

PARAMETERS = {'RandomForest': {'n_estimators': 50, 'n_jobs': -1},
              'Boosting': {'max_depth': 1, 'algorithm': "SAMME", 'n_estimators': 200},
              'LogisticRegression': {'penalty' : 'l1', 'C': 1e5},
              'SVM': {'kernel': 'linear', 'probability': 'True', 'random_state': 0},
              'DecisionTree': {},
              'KNeighbors': {'n_neighbors': 3},
            }



MODELS_TO_RUN = ['RandomForest', 'LogisticRegression', 'DecisionTree', 'SVM', 'Boosting', 'KNeighbors']


