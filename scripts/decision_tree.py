from scripts.classifier import Classifier
from sklearn import tree
import numpy as np


class DecisionTree(Classifier):
    name = "decision_tree.json"

    tuning_params = {'space': [(1, 5),
                               (1, 30),
                               (2, 20),
                               (1, 300)],
                     'n_calls': 100}

    def get_classifier(self):
        return tree.DecisionTreeClassifier()

    def set_params(self, clf, params):
        max_features, max_depth, min_samples_split, min_samples_leaf = params

        clf.set_params(max_features=max_features,
                       max_depth=max_depth,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf)

    def params_to_json(self, params):
        list = np.array(params).tolist()
        return {'max_features': list[0],
                'max_depth': list[1],
                'min_samples_split': list[2],
                'min_samples_leaf': list[3]}
