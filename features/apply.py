import time
import numpy as np
from features import temporal, tasks, freq, know


def get_users_posts_of_task1(list_of_K, users, posts):
    print('*** Get users and posts of task1 ***')
    start_time = time.time()
    users_of_task1, posts_of_task1 = {}, {}

    for K in list_of_K:
        posts_of_task1[K] = tasks.getTask1Posts(posts, K)
        users_of_task1[K] = tasks.getTask1Users(users, posts, K)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
    return users_of_task1, posts_of_task1


def prepare_features_of_task1(list_of_K, users_of_task1, posts):
    print('*** Prepare features of task1 ***')
    start_time = time.time()
    features_of_task1 = {}

    for K in list_of_K:
        features_of_task1[K] = tasks.getTask1Labels(users_of_task1[K], posts, K)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
    return features_of_task1


def get_users_posts_of_task2(list_of_T, users, posts):
    print('*** Get users and posts of task2 ***')
    start_time = time.time()
    users_of_task2, posts_of_task2 = {}, {}

    for T in list_of_T:
        posts_of_task2[T] = tasks.getTask2Posts(users, posts, T)
        users_of_task2[T] = tasks.getTask2Users(users, posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
    return users_of_task2, posts_of_task2


def prepare_features_of_task2(list_of_T, users_of_task2, posts):
    print('*** Prepare features of task2 ***')
    start_time = time.time()
    features_of_task2 = {}

    for T in list_of_T:
        features_of_task2[T] = tasks.getTask2Labels(users_of_task2[T], posts, T)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
    return features_of_task2


def temporal_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Temporal features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        features_of_task1[K]['gap1'] = temporal.getTimeGap1OfUser(users_of_task1[K], posts_of_task1[K])
        for k in range(2, K + 1):
            features_of_task1[K]['gap{}'.format(k)] = temporal.getTimeGapkOfPosts(posts_of_task1[K], k)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def temporal_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Temporal features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['gap1'] = temporal.getTimeGap1OfUser(users, posts)
        features_of_task2[T]['last_gap'] = temporal.getTimeLastGapOfPosts(posts).fillna(features_of_task2[T]['gap1'])
        features_of_task2[T]['time_since_last_post'] = temporal.getTimeSinceLastPost(users, posts, T)
        features_of_task2[T]['mean_gap'] = temporal.getTimeMeanGap(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def frequency_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Frequency features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        num_answers, num_questions = freq.getNumAnswers(posts), freq.getNumQuestions(posts)
        features_of_task1[K]['num_answers'] = num_answers
        features_of_task1[K]['num_questions'] = num_questions
        features_of_task1[K] = features_of_task1[K].fillna({'num_answers': 0, 'num_questions': 0})
        features_of_task1[K]['ans_que_ratio'] = freq.getAnsQuesRatio(num_answers, num_questions)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def frequency_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Frequency features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        num_answers, num_questions = freq.getNumAnswers(posts), freq.getNumQuestions(posts)
        features_of_task2[T]['num_answers'] = num_answers
        features_of_task2[T]['num_questions'] = num_questions
        features_of_task2[T] = features_of_task2[T].fillna({'num_answers': 0, 'num_questions': 0})
        features_of_task2[T]['ans_que_ratio'] = freq.getAnsQuesRatio(num_answers, num_questions)
        features_of_task2[T]['num_posts'] = freq.getNumPosts(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def knowledge_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1, posts_df):
    print('*** Knowledge features of task1 ***')
    start_time = time.time()

    # Extract knowledge features of task 1
    for K in list_of_K:
        print("Extract knowledge features of task1(K=", K, ")")
        users, posts = users_of_task1[K], posts_of_task1[K]
        answers, questions, qnta, tqna = know.prepareKnowledgeFeaturesOfTask1(users, posts, posts_df)
        features_of_task1[K]['accepted_answerer_rep'] =\
            know.getRepOfAcceptedAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['max_rep_answerer'] = know.getMaxRepAmongAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['num_que_answered'] = know.getNumQueAnswered(users, answers, questions, qnta, tqna)
        features_of_task1[K]['time_for_first_ans'] = know.getTimeForFirstAns(users, answers, questions, qnta, tqna)
        features_of_task1[K]['rep_questioner'] = know.getAvgRepOfQuestioner(users, answers, questions, qnta, tqna)
        features_of_task1[K]['rep_answerers'] = know.getAvgRepOfAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['rep_co_answerers'] = know.getAvgRepOfCoAnswerer(users, answers, questions, qnta, tqna)
        features_of_task1[K]['num_answers_recvd'] = know.getAvgNumAnsRecvd(users, answers, questions, qnta, tqna)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def knowledge_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Knowledge features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        print("Extract knowledge features of task2(T=)", T, ")")
        users, posts = users_of_task2[T], posts_of_task2[T]
        answers, questions, qna, qna1 = know.prepareKnowledgeFeaturesOfTask2(users, posts)
        features_of_task2[T]['accepted_answerer_rep'] =\
            know.getRepOfAcceptedAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['max_rep_answerer'] = know.getMaxRepAmongAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['num_que_answered'] = know.getNumQueAnswered(users, answers, questions, qna, qna1)
        features_of_task2[T]['time_for_first_ans'] = know.getTimeForFirstAns(users, answers, questions, qna, qna1)
        features_of_task2[T]['rep_questioner'] = know.getAvgRepOfQuestioner(users, answers, questions, qna, qna1)
        features_of_task2[T]['rep_answerers'] = know.getAvgRepOfAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['rep_co_answerers'] = know.getAvgRepOfCoAnswerer(users, answers, questions, qna, qna1)
        features_of_task2[T]['num_answers_recvd'] = know.getAvgNumAnsRecvd(users, answers, questions, qna, qna1)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def _fill_nan(features):
    if 'time_for_first_ans' in features.columns and np.isnan(features.time_for_first_ans).sum(0):
        features.time_for_first_ans = 1 / features.time_for_first_ans
        features.time_for_first_ans = features.time_for_first_ans.replace([np.nan], 0)

    fill_constants = {
        'accepted_answerer_rep': 0,
        'max_rep_answerer': 0,
        'num_que_answered': 0,
        'rep_questioner': 0,
        'rep_answerers': 0,
        'rep_co_answerers': 0,
        'num_answers_recvd': 0
    }
    return features.fillna(fill_constants)


def fill_nan(list_of_K, list_of_T, features_of_task1, features_of_task2):
    print('*** Fill NaN ***')
    start_time = time.time()

    for K in list_of_K:
        features_of_task1[K] = _fill_nan(features_of_task1[K])

    for T in list_of_T:
        features_of_task2[T] = _fill_nan(features_of_task2[T])

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
