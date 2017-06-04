from scripts.classyfier import Classifier
from sklearn import tree
from skopt import gp_minimize
from sklearn.metrics import mean_absolute_error
import numpy as np


class DecisionTree(Classifier):
    def __init__(self, data_list):
        Classifier.__init__(self, data_list)
        for _ in data_list:
            clf = tree.DecisionTreeClassifier()
            self._classifiers.append(clf)

    def objective(self, params):
        max_features, max_depth, min_samples_split, min_samples_leaf = params

        accuracy_list = list()
        for idx, clf in enumerate(self._classifiers):
            clf.set_params(
                max_features=max_features,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            )

            x = self.data_list[idx]['train'][:, :-1]
            y = self.data_list[idx]['train'][:, -1]
            clf.fit(x, y)

            prediction = clf.predict(self.data_list[idx]['valid'][:, :-1])

            accuracy_list.append(mean_absolute_error(self.data_list[idx]['valid'][:, -1], prediction))

        return -np.mean(accuracy_list)

    def tuning(self):
        space = [(1, 5),
                 (1, 5),
                 (2, 100),
                 (1, 100)]
        res_gp = gp_minimize(self.objective, space, n_calls=100, random_state=0)

        print("Best score=%.4f" % res_gp.fun)

        print("""Best parameters:
        - max_features=%d
        - max_depth=%.6f
        - min_samples_split=%d
        - min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1],
                                    res_gp.x[2], res_gp.x[3]))
