from scripts.read_data import read_all_data
from scripts.decision_tree import DecisionTree
from scripts.random_forest import RandomForest
from scripts.gradient_tree_boosting import GradientTreeBoosting


def experiment():
    data_list = read_all_data()
    classifiers_groups = [DecisionTree(data_list), RandomForest(data_list), GradientTreeBoosting(data_list)]

    results = []
    for classifier in classifiers_groups:
        classifier.tuning()
        classifier.learn()
        classifier.predict()
        results.append(classifier.results)

    return results

if __name__ == '__main__':
    results = experiment()
    print('finished')
