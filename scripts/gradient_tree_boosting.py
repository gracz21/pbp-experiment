from scripts.classyfier import Classifier
from sklearn.ensemble import GradientBoostingClassifier


class GradientTreeBoosting(Classifier):
    def __init__(self, data_list):
        Classifier.__init__(self, data_list)
        for _ in data_list:
            clf = GradientBoostingClassifier()
            self._classifiers.append(clf)

    def learn_single(self, idx):
        self._classifiers[idx].fit(self.data_list[idx]['train'].iloc[:, :-1], self.data_list[idx]['train'].iloc[:, -1])

    def tuning(self):
        pass
