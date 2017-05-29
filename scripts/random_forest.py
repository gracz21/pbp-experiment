from scripts.classyfier import Classifier
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Classifier):
    def __init__(self, data_list):
        Classifier.__init__(self, data_list)
        for _ in data_list:
            clf = RandomForestClassifier()
            self._classifiers.append(clf)

    def learn_single(self, idx):
        self._classifiers[idx].fit(self.data_list[idx]['train'].iloc[:, :-1], self.data_list[idx]['train'].iloc[:, -1])

    def tuning(self):
        pass
