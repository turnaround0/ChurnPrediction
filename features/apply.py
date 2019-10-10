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
