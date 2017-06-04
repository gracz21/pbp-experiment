from scripts.classyfier import Classifier
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Classifier):
    def __init__(self, data_list):
        Classifier.__init__(self, data_list)
        for _ in data_list:
            clf = RandomForestClassifier()
            self._classifiers.append(clf)

    def learn_single(self, idx):
        X = self.data_list[idx]['train'][:, :-1]
        Y = self.data_list[idx]['train'][:, -1]
        self._classifiers[idx].fit(X, Y)

    def tuning(self):
        pass
