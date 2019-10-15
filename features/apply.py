from features import temporal, tasks


def getFeatures(features, users, posts, task, K=None, T=None):
    assert (task in [1, 2])

    if -1 in features.index:
        features = features.drop([-1])

    return features


def apply_task1(users, posts):
    list_of_K = range(1, 21)
    users_of_task1, posts_of_task1 = {}, {}

    for K in list_of_K:
        posts_of_task1[K] = tasks.getTask1Posts(posts, K)
        users_of_task1[K] = tasks.getTask1Users(users, posts, K)

    return users_of_task1, posts_of_task1


def apply_pre_features_of_task1(users_of_task1, posts):
    list_of_K = range(1, 21)
    features_of_task1 = {}

    for K in list_of_K:
        features_of_task1[K] = tasks.getTask1Labels(users_of_task1[K], posts, K)

    return features_of_task1


def apply_task2(users, posts):
    list_of_T = [7, 15, 30]
    users_of_task2 = {}
    posts_of_task2 = {}

    for T in list_of_T:
        posts_of_task2[T] = tasks.getTask2Posts(users, posts, T)
        users_of_task2[T] = tasks.getTask2Users(users, posts)

    return users_of_task2, posts_of_task2


def apply_pre_features_of_task2(users_of_task2, posts):
    list_of_T = [7, 15, 30]
    features_of_task2 = {}

    for T in list_of_T:
        features_of_task2[T] = tasks.getTask2Labels(users_of_task2[T], posts, T)

    return features_of_task2


def apply_temporal_features_for_task1(features_of_task1, users_of_task1, posts_of_task1):
    list_of_K = range(1, 21)
    for K in list_of_K:
        features_of_task1[K]['gap1'] = temporal.getTimeGap1OfUser(users_of_task1[K], posts_of_task1[K])
        for k in range(2, K+1):
            features_of_task1[K]['gap{}'.format(k)] = temporal.getTimeGapkOfPosts(posts_of_task1[K], k)
