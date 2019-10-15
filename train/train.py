import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def init():
    seed = 1234
    np.random.seed(seed)


def LogisticRegression_(*arg, **kwarg):
    kwarg['max_iter'] = 1e3
    kwarg['solver'] = 'saga'
    kwarg['n_jobs'] = 8
    return LogisticRegression(*arg, **kwarg)


def learn_model(data, train_features, target='is_churn', model=DecisionTreeClassifier, seed=1234):
    X = data[train_features]
    y = data[target]
    print(model.__name__)

    ### 10-fold cross validation ###
    acc_list = []
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        ### Under-sampling ###
        churners = y_train[y_train == 1].index
        stayers = y_train[y_train == 0].index
        n_churn = len(churners)
        n_stay = len(stayers)
        if n_churn > n_stay:
            churners = np.random.choice(churners, n_stay, replace=False)
        else:
            stayers = np.random.choice(stayers, n_churn, replace=False)
        train_index = np.array(list(churners) + list(stayers))
        X_train, y_train = X.reindex(train_index), y.reindex(train_index)

        ### Learn Model ###
        mdl = model().fit(X_train, y_train)
        pred = mdl.predict(X_test)
        acc = (pred == y_test)
        acc_list.append(sum(acc) * 100 / len(acc))
    return acc_list


def table2(features_of_task1):
    # Table 2: Performance on Task 1
    drop_user_columns = ['Reputation', 'CreationDate', 'LastAccessDate', 'numPosts']

    # model = LogisticRegression_
    model = DecisionTreeClassifier

    list_of_K = range(1, 21)
    for K in list_of_K:
        print('Task 1, K={}'.format(K))
        train_features = [col for col in features_of_task1[K].columns
                          if col not in drop_user_columns + ['is_churn']]

        acc_list = learn_model(features_of_task1[K], train_features, model=model)
        print('Accuracy: {}'.format(np.mean(acc_list)))
        print('    for each folds: {}'.format(acc_list))


def table3(features_of_task2):
    # Table 3: Performance on Task 2
    drop_user_columns = ['Reputation', 'CreationDate', 'LastAccessDate']

    # model = LogisticRegression_
    model = DecisionTreeClassifier

    list_of_T = [7, 15, 30]
    for T in list_of_T:
        print('Task 2, T={}'.format(T))
        train_features = [col for col in features_of_task2[T].columns
                          if col not in drop_user_columns + ['is_churn']]

        acc_list = learn_model(features_of_task2[T], train_features, model=model)
        print('Accuracy: {}'.format(np.mean(acc_list)))
        print('    for each folds: {}'.format(acc_list))


def figure5(features_of_task1):
    # Figure 5: Churn prediction accuracy when features from each category are used in isolation
    temporal_features = ['gap1', 'last_gap', 'time_since_last_post', 'mean_gap']
    frequency_features = ['num_answers', 'num_questions',
                          'ans_que_ratio', 'num_posts']
    speed_features = ['answering_speed']
    quality_features = ['ans_score', 'que_score']
    consistency_features = ['ans_stddev', 'que_stddev']
    gratitude_features = ['ans_comments', 'que_comments']
    competitiveness_features = ['relative_rank_pos']
    content_features = ['ans_length', 'que_length']
    knowledge_features = ['accepted_answerer_rep', 'max_rep_answerer',
                          'num_que_answered', 'time_for_first_ans',
                          'rep_questioner', 'rep_answerers',
                          'rep_co_answerers', 'num_answers_recvd']

    analysis_feature_names = {
        'temporal': temporal_features,
        'frequency': frequency_features,
        'speed': speed_features,
        'quality': quality_features,
        'consistency': consistency_features,
        'gratitude': gratitude_features,
        'competitiveness': competitiveness_features,
        'content': content_features,
        'knowledge': knowledge_features,
    }

    task1_accuracy_of_category = {}
    for name, feature_list in analysis_feature_names.items():
        accuracy_of_category = []
        list_of_K = range(1, 21)
        for K in list_of_K:
            if name == 'temporal':
                feature_list = ['gap{}'.format(j) for j in range(1, K + 1)]
            elif name == 'frequency':
                features_list = [feat for feat in feature_list if feat != 'num_posts']
            train_features = [feat for feat in feature_list if feat in features_of_task1[K].columns]
            if len(train_features) == 0:
                continue
            print('\n{}, Task 1, K={}'.format(name, K))
            print('    columns: {}'.format(train_features))

            acc_list = learn_model(features_of_task1[K], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_of_category.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))
            print('    for each folds: {}'.format(acc_list))

        task1_accuracy_of_category[name] = accuracy_of_category

    # Bar Chart
    for title, predictions in task1_accuracy_of_category.items():
        if len(predictions) == 0:
            continue
        n_groups = len(list_of_K)
        index = np.arange(n_groups)

        plt.bar(index, predictions, tick_label=list_of_K, align='center')

        plt.title(title)
        plt.xlim(-1, n_groups)
        plt.ylim(40, 100)
        plt.show()


def temporal_feature_analysis(features_of_task1):
    ### Temporal Feature Analysis - Task 1 ###
    temporal_analysis_feature_func = {
        'gapK': lambda K: ['gap{}'.format(j) for j in range(1, K + 1)],
        'last_gap': lambda K: ['gap{}'.format(K)]
    }

    task1_accuracy_with_time_gap = {}
    list_of_K = range(1, 21)
    for K in list_of_K:
        accuracy_with_time_gap = []
        for name, feature_func in temporal_analysis_feature_func.items():
            feature_list = ['gap{}'.format(j) for j in range(1, K + 1)]
            train_features = [feat for feat in feature_list if feat in features_of_task1[K].columns]
            if len(train_features) == 0:
                continue
            print('\n{}, Task 1, K={}'.format(name, K))
            print('    columns: {}'.format(train_features))

            acc_list = learn_model(features_of_task1[K], train_features)
            mean_acc = np.mean(acc_list)
            accuracy_with_time_gap.append(mean_acc)
            print('Accuracy: {}'.format(mean_acc))
            print('    for each folds: {}'.format(acc_list))

        task1_accuracy_with_time_gap[K] = accuracy_with_time_gap

    # Table 4: Temporal gap features analysis
    for K, acc in task1_accuracy_with_time_gap.items():
        print(K, acc)