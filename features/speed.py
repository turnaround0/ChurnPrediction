# Speed features 1: answering_speed
def getAnsweringSpeed(posts):
    answers = posts[posts.PostTypeId == 2][['ParentId', 'CreationDate']]
    answers.columns = ['QuestionId', 'CreationDateA']
    questions = posts[posts.PostTypeId == 1][['OwnerUserId', 'CreationDate']]
    questions.columns = ['QuestionUserId', 'CreationDateQ']

    qna = answers.merge(questions, left_on='QuestionId', right_index=True)
    qna['ans_time'] = (qna.CreationDateA - qna.CreationDateQ).dt.total_seconds() / 60
    return 1 / qna.groupby('QuestionUserId').ans_time.mean()
