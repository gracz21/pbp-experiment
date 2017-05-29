from memory_profiler import memory_usage
from scripts.classyfier import Classifier
from sklearn.ensemble import GradientBoostingClassifier
from time import time


class GradientTreeBoosting(Classifier):
    def __init__(self, data_list):
        Classifier.__init__(self, data_list)
        for _ in data_list:
            clf = GradientBoostingClassifier()
            self._classifiers.append(clf)

    def learn(self):
        for idx, clf in enumerate(self._classifiers):
            mem_before = memory_usage(-1, interval=1, timeout=1)
            start_learn_time = time()
            mem_during = memory_usage(
                (clf.fit, (self.data_list[idx]['train'].iloc[:, :-1], self.data_list[idx]['train'].iloc[:, -1])))
            end_learn_time = time()
            mem_after = memory_usage(-1, interval=1, timeout=1)

            learn_time = end_learn_time - start_learn_time
            peak_memory = max(mem_during) - max(mem_before)
            memory = max(mem_after) - max(mem_before)

            self.results.append({'learn_time': learn_time, 'peak_memory': peak_memory, 'memory': memory})

    def tuning(self):
        pass
