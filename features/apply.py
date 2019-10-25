import time
import numpy as np
from features import temporal, tasks, freq, know, quality, consistency, speed,\
    gratitude, content, compet, answering, hot


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

    for K in list_of_K:
        print("Extract knowledge features of task1( K =", K, ")")
        users, posts = users_of_task1[K], posts_of_task1[K]
        answers, questions, qnta, tqna = know.prepareTask1(users, posts, posts_df)
        features_of_task1[K]['accepted_answerer_rep'] = know.getRepOfAcceptedAnswerer(qnta)
        features_of_task1[K]['max_rep_answerer'] = know.getMaxRepAmongAnswerer(qnta)
        features_of_task1[K]['num_que_answered'] = know.getNumQueAnswered(questions)
        features_of_task1[K]['time_for_first_ans'] = know.getTimeForFirstAns(questions, qnta)
        features_of_task1[K]['rep_questioner'] = know.getAvgRepOfQuestioner(tqna)
        features_of_task1[K]['rep_answerers'] = know.getAvgRepOfAnswerer(qnta)
        features_of_task1[K]['rep_co_answerers'] = know.getAvgRepOfCoAnswerer(users, answers, questions)
        features_of_task1[K]['num_answers_recvd'] = know.getAvgNumAnsRecvd(questions)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def knowledge_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Knowledge features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        print("Extract knowledge features of task2( T =", T, ")")
        users, posts = users_of_task2[T], posts_of_task2[T]
        answers, questions, qna, qna1 = know.prepareTask2(users, posts)
        features_of_task2[T]['accepted_answerer_rep'] = know.getRepOfAcceptedAnswerer(qna)
        features_of_task2[T]['max_rep_answerer'] = know.getMaxRepAmongAnswerer(qna)
        features_of_task2[T]['num_que_answered'] = know.getNumQueAnswered(questions)
        features_of_task2[T]['time_for_first_ans'] = know.getTimeForFirstAns(questions, qna)
        features_of_task2[T]['rep_questioner'] = know.getAvgRepOfQuestioner(qna1)
        features_of_task2[T]['rep_answerers'] = know.getAvgRepOfAnswerer(qna)
        features_of_task2[T]['rep_co_answerers'] = know.getAvgRepOfCoAnswerer(users, answers, questions)
        features_of_task2[T]['num_answers_recvd'] = know.getAvgNumAnsRecvd(questions)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def quality_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Quality features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        features_of_task1[K]['ans_score'] = quality.getScoreOfAnswers(posts)
        features_of_task1[K]['que_score'] = quality.getScoreOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def quality_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Quality features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['ans_score'] = quality.getScoreOfAnswers(posts)
        features_of_task2[T]['que_score'] = quality.getScoreOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def consistency_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Consistency features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        features_of_task1[K]['ans_stddev'] = consistency.getStdevOfScoresOfAnswers(posts)
        features_of_task1[K]['que_stddev'] = consistency.getStdevOfScoresOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def consistency_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Consistency features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['ans_stddev'] = consistency.getStdevOfScoresOfAnswers(posts)
        features_of_task2[T]['que_stddev'] = consistency.getStdevOfScoresOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def speed_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Speed features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        features_of_task1[K]['answering_speed'] = speed.getAnsweringSpeed(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def speed_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Speed features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['answering_speed'] = speed.getAnsweringSpeed(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def gratitude_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Gratitude features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        features_of_task1[K]['ans_comments'] = gratitude.getAvgNumOfAnswers(posts)
        features_of_task1[K]['que_comments'] = gratitude.getAvgNumOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def gratitude_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Gratitude features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['ans_comments'] = gratitude.getAvgNumOfAnswers(posts)
        features_of_task2[T]['que_comments'] = gratitude.getAvgNumOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def competitiveness_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Competitiveness features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        features_of_task1[K]['relative_rank_pos'] = compet.getRelRankPos(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def competitiveness_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Competitiveness features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['relative_rank_pos'] = compet.getRelRankPos(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def content_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1):
    print('*** Content features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        features_of_task1[K]['ans_length'] = content.getLengthOfAnswers(posts)
        features_of_task1[K]['que_length'] = content.getLengthOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def content_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Content features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        features_of_task2[T]['ans_length'] = content.getLengthOfAnswers(posts)
        features_of_task2[T]['que_length'] = content.getLengthOfQuestions(posts)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def answering_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1, posts_df):
    print('*** Answering features of task1 ***')
    start_time = time.time()

    for K in list_of_K:
        users, posts = users_of_task1[K], posts_of_task1[K]
        answers, questions, qnta, tqna = answering.prepareTask1(posts, posts_df)
        features_of_task1[K]['num_of_ans_count'] = answering.getAvgNumOfAnswerCount(tqna)
        features_of_task1[K]['first_post_type'] = answering.getFirstPostTypeIsAnswer(posts)
        features_of_task1[K]['total_comment'] = answering.getTotalNumOfComments(tqna)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def answering_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Answering features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        answers, questions, qna, qna1 = answering.prepareTask2(posts)
        features_of_task2[T]['num_of_ans_count'] = answering.getAvgNumOfAnswerCount(qna1)
        features_of_task2[T]['first_post_type'] = answering.getFirstPostTypeIsAnswer(posts)
        features_of_task2[T]['total_comment'] = answering.getTotalNumOfComments(qna1)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def hot_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2):
    print('*** Hot features of task2 ***')
    start_time = time.time()

    for T in list_of_T:
        users, posts = users_of_task2[T], posts_of_task2[T]
        questions, qna = hot.prepareTask2(posts)
        features_of_task2[T]['in_ans_hot_topic'] = hot.getNumAnswersInHotTopic(qna)
        features_of_task2[T]['in_ques_hot_topic'] = hot.getNumQuestionsInHotTopic(questions)
        features_of_task2[T]['in_hot_topic'] = hot.getNumInHotTopic(questions, qna)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def _fill_nan(features):
    # Cannot train NaN or infinite.
    if 'time_for_first_ans' in features.columns:
        features.time_for_first_ans = 1 / features.time_for_first_ans

    # In case of relative_rank_pos, its NaN should set to 0
    if 'relative_rank_pos' in features.columns:
        features.relative_rank_pos = features.relative_rank_pos.fillna(1)

    # All NaN is set to 0
    features = features.fillna(0)

    # All Inf is set to maximum value in the column
    features = features.replace([np.inf], np.nan)   # to remove when getting max
    for col in features.columns:
        features[col] = features[col].fillna(features[col].max())

    return features


def fill_nan(list_of_K, list_of_T, features_of_task1, features_of_task2):
    print('*** Fill NaN ***')
    start_time = time.time()

    for K in list_of_K:
        features_of_task1[K] = _fill_nan(features_of_task1[K])

    for T in list_of_T:
        features_of_task2[T] = _fill_nan(features_of_task2[T])

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
