import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import warnings

from train.train_config import training_models, analysis_feature_names, train_seed

warnings.filterwarnings("ignore", category=DeprecationWarning)


def random_init():
    np.random.seed(train_seed)


def do_under_sampling(y_train):
    churners = y_train[y_train == 1].index
    stayers = y_train[y_train == 0].index
    n_churn, n_stay = len(churners), len(stayers)

    # Before calling random.choice, random seed should set
    # for getting same dataset for each test
    random_init()
    # Maximum number of training samples is 20,000 due to training speed.
    num_max_choice = 10000
    num_choice = n_churn if n_churn < n_stay else n_stay
    num_choice = num_choice if num_choice < num_max_choice else num_max_choice

    churners = np.random.choice(churners, num_choice, replace=False)
    stayers = np.random.choice(stayers, num_choice, replace=False)

    return np.array(list(churners) + list(stayers))


def calc_stats(stats_list):
    tp, tn, fp, fn = stats_list
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 4)
    churn_accuracy = round(tp / (tp + fn), 4)
    stay_accuracy = round(tn / (tn + fp), 4)
    precision = round(tp / (tp + fp), 4)
    recall = round(tp / (tp + fn), 4)
    if recall == 0 and precision == 0:
        f1_score = 0
    else:
        f1_score = round(2 * (precision * recall) / (precision + recall), 4)

    print('Acc:', accuracy, 'Churn Acc:', churn_accuracy, 'Stay Acc:', stay_accuracy,
          'Precision:', precision, 'Recall:', recall, 'F1 score:', f1_score)

    return accuracy, churn_accuracy, stay_accuracy, precision, recall, f1_score


def get_avg_stats(stats_list):
    df_stats = pd.DataFrame(stats_list)
    df_stats.columns = ['tp', 'tn', 'fp', 'fn']

    tp = df_stats.tp.mean()
    tn = df_stats.tn.mean()
    fp = df_stats.fp.mean()
    fn = df_stats.fn.mean()

    return tp, tn, fp, fn


def learn_model(data, train_features, target='is_churn', model=DecisionTreeClassifier, seed=1234):
    # print('train_features:', train_features)
    X = data[train_features]
    y = data[target]
    start_time = time.time()

    # 10-fold cross validation
    acc_list, stats_list = [], []
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X):
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        train_index = do_under_sampling(y.iloc[train_index])
        X_train, y_train = X.reindex(train_index), y.reindex(train_index)

        # Learn Model
        mdl = model().fit(X_train, y_train)
        pred = mdl.predict(X_test)
        acc = (pred == y_test)
        acc_list.append(sum(acc) * 100 / len(acc))

        # Additional stats
        tp = ((pred == 1) & (y_test == 1)).sum()
        tn = ((pred == 0) & (y_test == 0)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        stats_list.append((tp, tn, fp, fn))

    end_time = time.time()
    print('Training time:', round(end_time - start_time, 8), 's')

    return acc_list, get_avg_stats(stats_list)


def store_stats(name, model_name, index_name, list_of_items, stats_model_list):
    df = pd.DataFrame(stats_model_list).rename_axis(index_name)
    df.columns = ['Acc', 'Churner Acc', 'Stayer Acc', 'Precision', 'Recall', 'F1 score']
    df.index = list_of_items
    df.to_csv('output/stats_on_' + name + '_' + model_name.lower().replace(' ', '_') + '.csv')


def performance_on_task1(list_of_K, features_of_task1):
    # Table 2: Performance on Task 1
    drop_user_columns = ['Id', 'Reputation', 'CreationDate', 'LastAccessDate', 'numPosts']
    acc_models = {}

    for model_name in training_models:
        model = training_models[model_name]
        acc_models[model_name] = {}
        stats_model_list = []
        print('\nTraining model name:', model_name)

        for K in list_of_K:
            train_features = [col for col in features_of_task1[K].columns
                              if col not in drop_user_columns + ['is_churn']]

            print('Task 1, K={}'.format(K))
            acc_list, stats_list = learn_model(features_of_task1[K], train_features, model=model)
            acc_mean = np.mean(acc_list)
            acc_models[model_name][K] = acc_mean
            print('Accuracy: {}'.format(acc_mean))
            stats_model_list.append(calc_stats(stats_list))

        store_stats('task1', model_name, 'K(posts)', list_of_K, stats_model_list)

    return acc_models


def performance_on_task2(list_of_T, features_of_task2):
    # Table 3: Performance on Task 2
    drop_user_columns = ['Id', 'Reputation', 'CreationDate', 'LastAccessDate', 'numPosts']
    acc_models = {}

    for model_name in training_models:
        model = training_models[model_name]
        acc_models[model_name] = {}
        stats_model_list = []
        print('\nTraining model name:', model_name)

        for T in list_of_T:
            train_features = [col for col in features_of_task2[T].columns
                              if col not in drop_user_columns + ['is_churn']]

            print('Task 2, T={}'.format(T))
            acc_list, stats_list = learn_model(features_of_task2[T], train_features, model=model)
            acc_mean = np.mean(acc_list)
            acc_models[model_name][T] = acc_mean
            print('Accuracy: {}'.format(acc_mean))
            stats_model_list.append(calc_stats(stats_list))

        store_stats('task2', model_name, 'T(days)', list_of_T, stats_model_list)

    return acc_models


def measure_task1_accuracy_of_category(list_of_K, features_of_task1):
    # Figure 5: Churn prediction accuracy when features from each category are used in isolation
    task1_accuracy_of_category = {}
    for name, feature_list in analysis_feature_names.items():
        accuracy_of_category = []
        for K in list_of_K:
            if name == 'Temporal':
                feature_list = ['gap{}'.format(j) for j in range(1, K + 1)]
            elif name == 'Frequency':
                feature_list = [feat for feat in feature_list if feat != 'num_posts']
            elif name == 'All':
                if K > 1:
                    feature_list = feature_list + ['gap{}'.format(K)]

            train_features = [feat for feat in feature_list if feat in features_of_task1[K].columns]
            if len(train_features) == 0:
                continue

            print('\n{}, Task 1, K={}'.format(name, K))
            acc_list, _ = learn_model(features_of_task1[K], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_of_category.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))

        task1_accuracy_of_category[name] = accuracy_of_category

    return task1_accuracy_of_category


def measure_task2_accuracy_of_category(list_of_T, features_of_task2):
    task2_accuracy_of_category = {}
    for name, feature_list in analysis_feature_names.items():
        accuracy_of_category = []
        for T in list_of_T:
            train_features = [feat for feat in feature_list if feat in features_of_task2[T].columns]
            if len(train_features) == 0:
                continue

            print('\n{}, Task 2, T={}'.format(name, T))
            acc_list, _ = learn_model(features_of_task2[T], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_of_category.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))

        task2_accuracy_of_category[name] = accuracy_of_category

    return task2_accuracy_of_category


def performance_on_temporal(list_of_K, features_of_task1):
    temporal_analysis_feature_func = {
        'gapK': lambda k: ['gap{}'.format(j) for j in range(1, k + 1)],
        'last_gap': lambda k: ['gap{}'.format(k)]
    }

    task1_accuracy_with_time_gap = {}
    for K in list_of_K:
        accuracy_with_time_gap = []
        for name, feature_func in temporal_analysis_feature_func.items():
            train_features = [feat for feat in feature_func(K) if feat in features_of_task1[K].columns]
            if len(train_features) == 0:
                continue

            print('\n{}, Task 1, K={}'.format(name, K))
            acc_list, _ = learn_model(features_of_task1[K], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_with_time_gap.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))

        task1_accuracy_with_time_gap[K] = accuracy_with_time_gap

    return task1_accuracy_with_time_gap
