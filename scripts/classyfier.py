from abc import ABC, abstractmethod
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score, f1_score
from time import time


class Classifier(ABC):
    def __init__(self, data_list):
        self.data_list = data_list
        self._classifiers = []
        self.results = []

    @abstractmethod
    def tuning(self):
        pass

    def learn(self):
        for idx, _ in enumerate(self._classifiers):
            mem_before = memory_usage(-1, interval=1, timeout=1)
            start_learn_time = time()
            self.learn_single(idx)
            end_learn_time = time()
            mem_after = memory_usage(-1, interval=1, timeout=1)

            learn_time = end_learn_time - start_learn_time

            self.results.append({'learn_time': learn_time})

    @abstractmethod
    def learn_single(self, idx):
        pass

    def predict(self):
        for idx, clf in enumerate(self._classifiers):
            start_predict_time = time()
            prediction = clf.predict(self.data_list[idx]['test'][:, :-1])
            end_predict_time = time()

            accuracy = accuracy_score(self.data_list[idx]['test'][:, -1], prediction)
            
            if self.data_list[idx]['classes'] == 2:
                f_score = f1_score(self.data_list[idx]['test'][:, -1], prediction)
            else:
                f_score = -1

            predict_time = end_predict_time - start_predict_time

            self.results[idx].update({'predict_time': predict_time, 'accuracy': accuracy, 'f_score': f_score})
