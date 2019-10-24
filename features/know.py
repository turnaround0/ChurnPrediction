# For the fast extraction, prepare questions x answers
def prepareTask1(users, posts, all_posts):
    answers = posts[posts.PostTypeId == 2][['OwnerUserId', 'ParentId', 'CreationDate']]
    answers.columns = ['AnswerUserId', 'QuestionId', 'CreationDateA']
    all_answers = all_posts[all_posts.PostTypeId == 2][['OwnerUserId', 'ParentId', 'CreationDate']]
    all_answers.columns = ['AnswerUserId', 'QuestionId', 'CreationDateA']

    questions = posts[posts.PostTypeId == 1][
        ['OwnerUserId', 'AcceptedAnswerId', 'AnswerCount', 'CreationDate']
    ].rename(columns={'OwnerUserId': 'QuestionUserId', 'CreationDate': 'CreationDateQ'})
    all_questions = all_posts[all_posts.PostTypeId == 1][
        ['OwnerUserId', 'AcceptedAnswerId', 'AnswerCount', 'CreationDate']
    ].rename(columns={'OwnerUserId': 'QuestionUserId', 'CreationDate': 'CreationDateQ'})

    # Questions and Total Answers
    qnta = all_answers.merge(questions, left_on='QuestionId', right_index=True)\
        .merge(users[['Reputation']], left_on='AnswerUserId', right_index=True)
    # Total Questions and Answers
    tqna = answers.merge(all_questions, left_on='QuestionId', right_index=True)\
        .merge(users[['Reputation']], left_on='QuestionUserId', right_index=True)

    return answers, questions, qnta, tqna


# For the fast extraction, prepare questions x answers
def prepareTask2(users, posts):
    answers = posts[posts.PostTypeId == 2][['OwnerUserId', 'ParentId', 'CreationDate']]
    answers.columns = ['AnswerUserId', 'QuestionId', 'CreationDateA']
    questions = posts[posts.PostTypeId == 1][['OwnerUserId', 'AcceptedAnswerId', 'AnswerCount', 'CreationDate']]\
        .rename(columns={'OwnerUserId': 'QuestionUserId', 'CreationDate': 'CreationDateQ'})

    ans_ques = answers.merge(questions, left_on='QuestionId', right_index=True)
    qna = ans_ques.merge(users[['Reputation']], left_on='AnswerUserId', right_index=True)
    qna2 = ans_ques.merge(users[['Reputation']], left_on='QuestionUserId', right_index=True)

    return answers, questions, qna, qna2


# Knowledge features 1: accepted_answerer_rep
def getRepOfAcceptedAnswerer(qnta):
    return qnta[qnta.AcceptedAnswerId == qnta.index].groupby('QuestionUserId').Reputation.max()


# Knowledge features 2: max_rep_answerer
def getMaxRepAmongAnswerer(qnta):
    return qnta.groupby('QuestionUserId').Reputation.max()


# Knowledge features 3: num_que_answered
def getNumQueAnswered(questions):
    # number of questions posted by the user that got answered
    return questions[questions.AnswerCount > 0].groupby('QuestionUserId').size()


# Knowledge features 4: time_for_first_ans
def getTimeForFirstAns(questions, qnta):
    tmp = qnta[qnta.CreationDateQ < qnta.CreationDateA]
    tmp['time_for_ans'] = (tmp.CreationDateA - tmp.CreationDateQ).dt.total_seconds() / 60
    questions['time_for_first_ans'] = tmp.groupby('QuestionId').time_for_ans.min()
    return questions.groupby('QuestionUserId').time_for_first_ans.mean()


# Knowledge features 5: rep_questioner
def getAvgRepOfQuestioner(tqna):
    # Avg. reputation of the user whose question was answered
    return tqna.groupby('AnswerUserId').Reputation.mean()


# Knowledge features 6: rep_answerers
def getAvgRepOfAnswerer(qnta):
    # Avg. reputation of the users who answered the question
    return qnta.groupby('QuestionUserId').Reputation.mean()


# Knowledge features 7: rep_co_answerers
def getAvgRepOfCoAnswerer(users, answers, questions):
    rep_ans = answers.merge(questions, left_on='QuestionId', right_index=True)\
        .merge(users[['Reputation']], left_on='AnswerUserId', right_index=True)
    avg_rep_ans = rep_ans.groupby('QuestionId').Reputation.mean()
    rep_co_answerer = answers.merge(avg_rep_ans, left_on='QuestionId', right_index=True)
    return rep_co_answerer.groupby('AnswerUserId').Reputation.mean()


# Knowledge features 8: num_answers_received
def getAvgNumAnsRecvd(questions):
    return questions.groupby('QuestionUserId').AnswerCount.mean()
