# Speed features 1: answering_speed
def getAnsweringSpeed(posts):
    answers = posts[posts.PostTypeId == 2][['ParentId', 'CreationDate']]
    answers.columns = ['QuestionId', 'CreationDateA']
    questions = posts[posts.PostTypeId == 1][['OwnerUserId', 'CreationDate']]
    questions.columns = ['QuestionUserId', 'CreationDateQ']

    qna = answers.merge(questions, left_on='QuestionId', right_index=True)
    qna['inv_ans_time'] = 1 / ((qna.CreationDateA - qna.CreationDateQ).dt.total_seconds() / 60)
    return qna.groupby('QuestionUserId').inv_ans_time.mean()
