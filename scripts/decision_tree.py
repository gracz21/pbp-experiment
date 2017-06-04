from scripts.classyfier import Classifier
from sklearn import tree


class DecisionTree(Classifier):
    def __init__(self, data_list):
        Classifier.__init__(self, data_list)
        for _ in data_list:
            clf = tree.DecisionTreeClassifier()
            self._classifiers.append(clf)

    def tuning(self):
        pass

    def learn_single(self, idx):
        X = self.data_list[idx]['train'][:, :-1]
        Y = self.data_list[idx]['train'][:, -1]
        self._classifiers[idx].fit(X, Y)
