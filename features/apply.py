from features import temporal, tasks, freq, know


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


def apply_temporal_features_for_task2(features_of_task2, users_of_task2, posts_of_task2):
    list_of_T = [7, 15, 30]
    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['gap1'] = temporal.getTimeGap1OfUser(users, posts)
        features_of_task2[T]['last_gap'] = temporal.getTimeLastGapOfPosts(posts).fillna(features_of_task2[T]['gap1'])
        # features_of_task2[T]['last_gap'] = temporal.getTimeLastGapOfPosts(posts).fillna(0)
        features_of_task2[T]['time_since_last_post'] = temporal.getTimeSinceLastPost(users, posts, T)
        features_of_task2[T]['mean_gap'] = temporal.getTimeMeanGap(posts)


def apply_frequency_features_of_task1(features_of_task1, users_of_task1, posts_of_task1):
    list_of_K = range(1, 21)
    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        features_of_task1[K]['num_answers'] = freq.getNumAnswers(posts)
        features_of_task1[K]['num_questions'] = freq.getNumQuestions(posts)
        features_of_task1[K] = features_of_task1[K].fillna({'num_answers': 0, 'num_questions': 0})
        features_of_task1[K]['ans_que_ratio'] = \
            freq.getAnsQuesRatio(features_of_task1[K]['num_answers'], features_of_task1[K]['num_questions'])


def apply_frequency_features_of_task2(features_of_task2, users_of_task2, posts_of_task2):
    list_of_T = [7, 15, 30]
    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['num_answers'] = freq.getNumAnswers(posts)
        features_of_task2[T]['num_questions'] = freq.getNumQuestions(posts)
        features_of_task2[T] = features_of_task2[T].fillna({'num_answers': 0, 'num_questions': 0})
        features_of_task2[T]['ans_que_ratio'] = \
            freq.getAnsQuesRatio(features_of_task2[T]['num_answers'], features_of_task2[T]['num_questions'])
        features_of_task2[T]['num_posts'] = freq.getNumPosts(posts)


def apply_knowledge_features_of_task1(features_of_task1, users_of_task1, posts_of_task1, posts_df):
    # Extract knowledge features of task 1
    list_of_K = range(1, 21)
    for K in list_of_K:
        print("Extract knowledge features of task1(K=", K, ")")
        users, posts = users_of_task1[K], posts_of_task1[K]
        answers, questions, qnta, tqna = know.preprocessForKnowledgeFeaturesForTask1(users, posts, posts_df)
        features_of_task1[K]['accepted_answerer_rep'] = know.getRepOfAcceptedAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['max_rep_answerer'] = know.getMaxRepAmongAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['num_que_answered'] = know.getNumQueAnswered(users, answers, questions, qnta, tqna)
        features_of_task1[K]['time_for_first_ans'] = know.getTimeForFirstAns(users, answers, questions, qnta, tqna)
        features_of_task1[K]['rep_questioner'] = know.getAvgRepOfQuestioner(users, answers, questions, qnta, tqna)
        features_of_task1[K]['rep_answerers'] = know.getAvgRepOfAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['rep_co_answerers'] = know.getAvgRepOfCoAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['num_answers_recvd'] = know.getAvgNumAnsReceived(users, answers, questions, qnta, tqna)
        print(features_of_task1[K].accepted_answerer_rep)


def apply_knowledge_features_of_task2(features_of_task2, users_of_task2, posts_of_task2):
    list_of_T = [7, 15, 30]
    for T in list_of_T:
        print("Extract knowledge features of task2(T=)", T, ")")
        users, posts = users_of_task2[T], posts_of_task2[T]
        answers, questions, qna, qna1 = know.preprocessForKnowledgeFeaturesForTask2(users, posts)
        features_of_task2[T]['accepted_answerer_rep'] = know.getRepOfAcceptedAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['max_rep_answerer'] = know.getMaxRepAmongAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['num_que_answered'] = know.getNumQueAnswered(users, answers, questions, qna, qna1)
        features_of_task2[T]['time_for_first_ans'] = know.getTimeForFirstAns(users, answers, questions, qna, qna1)
        features_of_task2[T]['rep_questioner'] = know.getAvgRepOfQuestioner(users, answers, questions, qna, qna1)
        features_of_task2[T]['rep_answerers'] = know.getAvgRepOfAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['rep_co_answerers'] = know.getAvgRepOfCoAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['num_answers_recvd'] = know.getAvgNumAnsReceived(users, answers, questions, qna, qna1)
