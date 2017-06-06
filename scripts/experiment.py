from scripts.read_data import read_all_data
from scripts.decision_tree import DecisionTree
from scripts.random_forest import RandomForest
from scripts.gradient_tree_boosting import GradientTreeBoosting
from time import gmtime, strftime
from tabulate import tabulate
import csv


def experiment():
    data_list = read_all_data()
    classifiers_groups = [DecisionTree(data_list), RandomForest(data_list), GradientTreeBoosting(data_list)]

    results = {}
    for classifier in classifiers_groups:
        classifier.create_classifiers()
        print(classifier.tuning())
        classifier.learn()
        classifier.predict()
        results[classifier.name] = classifier.results

    return results

if __name__ == '__main__':
    results = experiment()
    for key, val in results.items():
        filename_base = './results/' + key + '_' + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        with open(filename_base + '.tex', 'w') as f:
            f.write(tabulate(val, headers='keys', tablefmt="latex", floatfmt=".2f"))

        with open(filename_base + '.csv', 'w') as f:
            w = csv.DictWriter(f, list(val[0].keys()), lineterminator='\n', delimiter=';')
            w.writeheader()
            w.writerows(val)
