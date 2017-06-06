from scripts.classifier import Classifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


class GradientTreeBoosting(Classifier):
    name = "gradient_tree_boosting"

    tuning_params = {'space': [(1, 30),
                               (2, 20),
                               (1, 300),
                               (5, 200),
                               (0.05, 0.2)],
                     'n_calls': 20}

    def get_classifier(self):
        return GradientBoostingClassifier()

    def set_params(self, clf, params):
        max_depth, min_samples_split, min_samples_leaf, n_estimators, learning_rate = params

        clf.set_params(max_features='sqrt',
                       max_depth=max_depth,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf,
                       n_estimators=n_estimators,
                       learning_rate=learning_rate)

    def params_to_json(self, params):
        list = np.array(params).tolist()
        return {'max_features': 'sqrt',
                'max_depth': list[0],
                'min_samples_split': list[1],
                'min_samples_leaf': list[2],
                'n_estimators': list[3],
                'learning_rate': list[4]}
