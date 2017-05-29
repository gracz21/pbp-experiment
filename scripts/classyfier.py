from abc import ABC, abstractmethod
from time import time
from sklearn.metrics import accuracy_score, f1_score


class Classifier(ABC):
    _classifiers = []
    results = []

    def __init__(self, data_list):
        self.data_list = data_list

    @abstractmethod
    def tuning(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    def predict(self):
        for idx, clf in enumerate(self._classifiers):
            start_predict_time = time()
            prediction = clf.predict(self.data_list[idx]['test'].iloc[:, :-1])
            end_predict_time = time()

            accuracy = accuracy_score(self.data_list[idx]['test'].iloc[:, -1], prediction)
            f_score = f1_score(self.data_list[idx]['test'].iloc[:, -1], prediction)
            predict_time = end_predict_time - start_predict_time

            self.results[idx].update({'predict_time': predict_time, 'accuracy': accuracy, 'f_score': f_score})
