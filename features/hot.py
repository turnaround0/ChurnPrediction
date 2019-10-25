def prepareTask2(posts):
    questions = posts[posts.PostTypeId == 1][['AnswerCount', 'OwnerUserId']] \
        .rename(columns={'OwnerUserId': 'QuestionUserId'})
    answers = posts[posts.PostTypeId == 2][['ParentId', 'OwnerUserId']] \
        .rename(columns={'OwnerUserId': 'AnswerUserId'})
    hot_answer_count = questions.AnswerCount.quantile(0.95)

    hot_questions = questions[questions.AnswerCount >= hot_answer_count]
    hot_qna = hot_questions.merge(answers, left_index=True, right_on='ParentId')

    return hot_questions, hot_qna


# Hot features 1: in_hot_topic
def getNumInHotTopic(questions, qna):
    return qna.groupby('AnswerUserId').size().add(questions.groupby('QuestionUserId').size(), fill_value=0)


def getNumQuestionsInHotTopic(questions):
    return questions.groupby('QuestionUserId').size()


def getNumAnswersInHotTopic(qna):
    return qna.groupby('AnswerUserId').size()
