from scripts.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForest(Classifier):
    name = "random_forest"

    tuning_params = {'space': [(1, 30),
                               (2, 20),
                               (1, 300),
                               (5, 30)],
                     'n_calls': 20}

    def get_classifier(self):
        return RandomForestClassifier()

    def set_params(self, clf, params):
        max_depth, min_samples_split, min_samples_leaf, n_estimators = params

        clf.set_params(max_features='sqrt',
                       max_depth=max_depth,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf,
                       n_estimators=n_estimators)

    def params_to_json(self, params):
        list = np.array(params).tolist()
        return {'max_features': 'sqrt',
                'max_depth': list[0],
                'min_samples_split': list[1],
                'min_samples_leaf': list[2],
                'n_estimators': list[3]}
