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
        self._classifiers[idx].fit(self.data_list[idx]['train'].iloc[:, :-1], self.data_list[idx]['train'].iloc[:, -1])
