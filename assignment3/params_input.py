'''
Parameters for 
'''


CLASSIFIERS = {
            'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
            'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            'LR': LogisticRegression(penalty='l1', C=1e5),
            'SVM': SVC(kernel='linear', probability=True, random_state=0),
            'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(),
            'SGD': SGDClassifier(loss="hinge", penalty="l2"),
            'KNN': KNeighborsClassifier(n_neighbors=3)
            }


TEST_GRID = {
            'RF': {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'LR': {'penalty': ['l1'], 'C': [0.01]},
            'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
            'ET': {'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'AB': {'algorithm': ['SAMME'], 'n_estimators': [1]},
            'GB': {'n_estimators': [1], 'learning_rate' : [0.1], 'subsample' : [0.5], 'max_depth': [1]},
            'NB': {},
            'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
            'SVM': {'C':[0.01],'kernel':['linear']},
            'KNN': {'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
       }
