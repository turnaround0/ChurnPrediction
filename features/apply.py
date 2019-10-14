from features.tasks import getTask1Posts, getTask1Users, getTask2Posts, getTask2Users, prepareFeaturesTask1, \
    prepareFeaturesTask2


def getFeatures(features, users, posts, task, K=None, T=None):
    assert (task in [1, 2])

    if -1 in features.index:
        features = features.drop([-1])

    return features


"""
task1_features = []
for K in range(1, 20 + 1):
    task1_features.append()

task2_features = []
for T in [7, 15, 30]:
    task2_features.append()
"""


def apply_task1(users, posts):
    list_of_K = range(1, 21)
    users_of_task1, posts_of_task1 = {}, {}

    for K in list_of_K:
        posts_of_task1[K] = getTask1Posts(posts, K)
        users_of_task1[K] = getTask1Users(users, posts, K)

    return users_of_task1, posts_of_task1


def apply_pre_features_of_task1(users_of_task1, posts):
    list_of_K = range(1, 21)
    features_of_task1 = {}

    for K in list_of_K:
        features_of_task1[K] = prepareFeaturesTask1(users_of_task1[K], posts, K)

    return features_of_task1


def apply_task2(users, posts):
    list_of_T = [7, 15, 30]
    users_of_task2 = {}
    posts_of_task2 = {}

    for T in list_of_T:
        posts_of_task2[T] = getTask2Posts(users, posts, T)
        users_of_task2[T] = getTask2Users(users, posts)

    return users_of_task2, posts_of_task2


def apply_pre_features_of_task2(users_of_task2, posts):
    list_of_T = [7, 15, 30]
    features_of_task2 = {}

    for T in list_of_T:
        features_of_task2[T] = prepareFeaturesTask2(users_of_task2[T], posts, T)

    return features_of_task2
