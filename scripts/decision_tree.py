from memory_profiler import memory_usage
from scripts import read_data
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score
from time import time


def main():
    data_list = read_data.read_all_data()
    classifiers = []
    results = []
    for _ in data_list:
        clf = tree.DecisionTreeClassifier()
        classifiers.append(clf)

    # tuning

    for idx, clf in enumerate(classifiers):
        mem_before = memory_usage(-1, interval=.2, timeout=1)
        start_learn_time = time()
        mem_during = memory_usage(
            (clf.fit, (data_list[idx]['train'].iloc[:, :-1], data_list[idx]['train'].iloc[:, -1])))
        end_learn_time = time()

        start_predict_time = time()
        prediction = clf.predict(data_list[idx]['test'].iloc[:, :-1])
        end_predict_time = time()

        learn_time = end_learn_time - start_learn_time
        predict_time = end_predict_time - start_predict_time
        memory = max(mem_during) - max(mem_before)
        accuracy = accuracy_score(data_list[idx]['test'].iloc[:, -1], prediction)
        f_score = f1_score(data_list[idx]['test'].iloc[:, -1], prediction)

        results.append({'learn_time': learn_time, 'predict_time': predict_time, 'memory': memory,
                        'accuracy': accuracy, 'f_score': f_score})

if __name__ == '__main__':
    main()
