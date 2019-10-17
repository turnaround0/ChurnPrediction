import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import warnings

from train.train_config import training_models, analysis_feature_names, train_seed

warnings.filterwarnings("ignore", category=DeprecationWarning)


def init():
    np.random.seed(train_seed)


def do_under_sampling(y_train):
    churners = y_train[y_train == 1].index
    stayers = y_train[y_train == 0].index
    n_churn, n_stay = len(churners), len(stayers)

    if n_churn > n_stay:
        churners = np.random.choice(churners, n_stay, replace=False)
    else:
        stayers = np.random.choice(stayers, n_churn, replace=False)

    return np.array(list(churners) + list(stayers))


def learn_model(data, train_features, target='is_churn', model=DecisionTreeClassifier, seed=1234):
    X = data[train_features]
    y = data[target]

    start_time = time.time()

    # 10-fold cross validation
    acc_list = []
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

    end_time = time.time()
    print('Training time:', round(end_time - start_time, 8), 's')

    return acc_list


def performance_on_task1(list_of_K, features_of_task1):
    # Table 2: Performance on Task 1
    drop_user_columns = ['Reputation', 'CreationDate', 'LastAccessDate', 'numPosts']
    acc_models = {}

    for model_name in training_models:
        model = training_models[model_name]
        acc_models[model_name] = {}
        print('Training model name:', model_name)

        for K in list_of_K:
            train_features = [col for col in features_of_task1[K].columns
                              if col not in drop_user_columns + ['is_churn']]

            print('Task 1, K={}'.format(K))
            acc_list = learn_model(features_of_task1[K], train_features, model=model)
            acc_mean = np.mean(acc_list)
            print('Accuracy: {}'.format(acc_mean))
            # print('    for each folds: {}'.format(acc_list))
            acc_models[model_name][K] = acc_mean

    return acc_models


def performance_on_task2(list_of_T, features_of_task2):
    # Table 3: Performance on Task 2
    drop_user_columns = ['Reputation', 'CreationDate', 'LastAccessDate', 'numPosts']
    acc_models = {}

    for model_name in training_models:
        model = training_models[model_name]
        acc_models[model_name] = {}
        print('Training model name:', model_name)

        for T in list_of_T:
            train_features = [col for col in features_of_task2[T].columns
                              if col not in drop_user_columns + ['is_churn']]

            print('Task 2, T={}'.format(T))
            acc_list = learn_model(features_of_task2[T], train_features, model=model)
            acc_mean = np.mean(acc_list)
            print('Accuracy: {}'.format(acc_mean))
            # print('    for each folds: {}'.format(acc_list))
            acc_models[model_name][T] = acc_mean

    return acc_models


def measure_task1_accuracy_of_category(list_of_K, features_of_task1):
    # Figure 5: Churn prediction accuracy when features from each category are used in isolation
    task1_accuracy_of_category = {}
    for name, feature_list in analysis_feature_names.items():
        accuracy_of_category = []
        for K in list_of_K:
            if name == 'temporal':
                feature_list = ['gap{}'.format(j) for j in range(1, K + 1)]
            elif name == 'frequency':
                feature_list = [feat for feat in feature_list if feat != 'num_posts']
            train_features = [feat for feat in feature_list if feat in features_of_task1[K].columns]
            if len(train_features) == 0:
                continue
            print('\n{}, Task 1, K={}'.format(name, K))
            # print('    columns: {}'.format(train_features))

            acc_list = learn_model(features_of_task1[K], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_of_category.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))
            # print('    for each folds: {}'.format(acc_list))

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
            # print('    columns: {}'.format(train_features))

            acc_list = learn_model(features_of_task2[T], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_of_category.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))
            # print('    for each folds: {}'.format(acc_list))

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
            feature_list = ['gap{}'.format(j) for j in range(1, K + 1)]
            train_features = [feat for feat in feature_list if feat in features_of_task1[K].columns]
            if len(train_features) == 0:
                continue
            print('\n{}, Task 1, K={}'.format(name, K))
            # print('    columns: {}'.format(train_features))

            acc_list = learn_model(features_of_task1[K], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_with_time_gap.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))
            # print('    for each folds: {}'.format(acc_list))

        task1_accuracy_with_time_gap[K] = accuracy_with_time_gap

    return task1_accuracy_with_time_gap
