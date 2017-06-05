from abc import ABC, abstractmethod
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from time import time
import json
from skopt import gp_minimize
import numpy as np

class Classifier(ABC):
    @property
    def name(self):
        raise NotImplementedError


    @property
    def tuning_params(self):
        raise NotImplementedError


    @abstractmethod
    def set_params(self, clf, params):
        raise NotImplementedError


    @abstractmethod
    def get_classifier(self):
        raise NotImplementedError


    def __init__(self, data_list):
        self.data_list = data_list
        self.classifiers = []
        self.results = []


    def create_classifiers(self):
        for _ in self.data_list:
            clf = self.get_classifier()
            self.classifiers.append(clf)


    def tuning(self):
        res_gp = gp_minimize(self.objective, self.tuning_params['space'], 
                             n_calls=self.tuning_params['n_calls'], verbose=True)
        params = self.params_to_json(res_gp.x)
        self.save_params(params)
        return res_gp.fun


    def objective(self, params):
        accuracy_list = list()
        for idx, clf in enumerate(self.classifiers):
            self.set_params(clf, params)

            x = self.data_list[idx]['train'].iloc[:, :-1]
            y = self.data_list[idx]['train'].iloc[:, -1]
            clf.fit(x, y)

            prediction = clf.predict(self.data_list[idx]['valid'].iloc[:, :-1])
            accuracy_list.append(mean_absolute_error(self.data_list[idx]['valid'].iloc[:, -1], prediction))

        return -np.mean(accuracy_list)


    def load_params(self):
        with open('./params/' + self.name) as f:
            return json.load(f)

    def save_params(self, params):
        with open('./params/' + self.name + '.json', 'w') as f:
            json.dump(params, f, sort_keys=False, indent=4, separators=(',', ': '))


    def learn(self):
        params = self.load_params()
        for idx, clf in enumerate(self.classifiers):
            clf.set_params(**params)
            x = self.data_list[idx]['train'].iloc[:, :-1]
            y = self.data_list[idx]['train'].iloc[:, -1]

            mem_before = memory_usage(-1, interval=1, timeout=1)
            start_learn_time = time()
            mem_during = memory_usage((clf.fit, (x, y)))
            end_learn_time = time()
            mem_after = memory_usage(-1, interval=1, timeout=1)

            learn_time = end_learn_time - start_learn_time
            peak_memory = max(mem_during) - max(mem_before)
            memory = max(mem_after) - max(mem_before)

            self.results.append({'name': self.name, 'learn_time': learn_time, 'peak_memory': peak_memory, 'memory': memory})


    def predict(self):
        for idx, clf in enumerate(self.classifiers):
            start_predict_time = time()
            prediction = clf.predict(self.data_list[idx]['test'].iloc[:, :-1])
            end_predict_time = time()

            accuracy = accuracy_score(self.data_list[idx]['test'].iloc[:, -1], prediction)

            if self.data_list[idx]['classes'] == 2:
                f_score = f1_score(self.data_list[idx]['test'].iloc[:, -1], prediction)
            else:
                f_score = f1_score(self.data_list[idx]['test'].iloc[:, -1], prediction, average="macro")

            predict_time = end_predict_time - start_predict_time

            self.results[idx].update({'predict_time': predict_time, 'accuracy': accuracy, 'f_score': f_score})
