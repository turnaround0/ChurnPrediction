from train import train_models

training_models = {
    'Decision Tree': train_models.decision_tree_classifier_with_args,
    'Tree extraction': train_models.decision_tree_ext_method,
    # 'SVM (Linear)': train_models.linear_svc_with_args,
    # 'SVM (RBF)': train_models.svc_with_args,
    # 'Logistic Regression': train_models.logistic_regression_with_args,
}

temporal_features = ['gap1', 'last_gap', 'time_since_last_post', 'mean_gap']
frequency_features = ['num_answers', 'num_questions', 'ans_que_ratio', 'num_posts']
speed_features = ['answering_speed']
quality_features = ['ans_score', 'que_score']
consistency_features = ['ans_stddev', 'que_stddev']
gratitude_features = ['ans_comments', 'que_comments']
competitiveness_features = ['relative_rank_pos']
content_features = ['ans_length', 'que_length']
knowledge_features = ['accepted_answerer_rep', 'max_rep_answerer', 'num_que_answered', 'time_for_first_ans',
                      'rep_questioner', 'rep_answerers', 'rep_co_answerers', 'num_answers_recvd']
all_features = consistency_features + gratitude_features + frequency_features + quality_features\
               + competitiveness_features + speed_features + content_features + knowledge_features + temporal_features

analysis_feature_names = {
    'Consistency': consistency_features,
    'Gratitude': gratitude_features,
    'Frequency': frequency_features,
    'Quality': quality_features,
    'Competitiveness': competitiveness_features,
    'Speed': speed_features,
    'Content': content_features,
    'Knowledge': knowledge_features,
    'Temporal': temporal_features,
    'All': all_features
}

answering_features = ['num_of_ans_count', 'first_post_type', 'total_comment']
answering_features_1 = ['num_of_ans_count']
answering_features_2 = ['first_post_type']
answering_features_3 = ['total_comment']

hot_features = ['in_hot_topic', 'in_ans_hot_topic', 'in_ques_hot_topic']
hot_features_1 = ['in_hot_topic']
hot_features_2 = ['in_ans_hot_topic']
hot_features_3 = ['in_ques_hot_topic']
hot_features_4 = ['num_answers']

additional_features = {
    'Answering': answering_features,
    'Answering_num_of_ans_count': answering_features_1,
    'Answering_first_post_type': answering_features_2,
    'Answering_total_comment': answering_features_3,
    'Hot': hot_features,
    'Hot_in_hot_topic': hot_features_1,
    'Hot_in_ans_hot_topic': hot_features_2,
    'Hot_in_ques_hot_topic': hot_features_3,
    'Hot_num_answers': hot_features_4,
}

train_seed = 1234
