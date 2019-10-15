# For the fast extraction, prepare questions x answers
def preprocessForKnowledgeFeaturesForTask1(users, posts, all_posts):
    answers = posts[posts.Answer == 1].reset_index()
    questions = posts[posts.Question == 1].reset_index()
    all_answers = all_posts[all_posts.Answer == 1].reset_index()
    all_questions = all_posts[all_posts.Question == 1].reset_index()

    qnta = all_answers.set_index('ParentId').join(questions, how='inner', lsuffix='A', rsuffix='Q')
    tqna = answers.set_index('ParentId').join(all_questions, how='inner', lsuffix='A', rsuffix='Q')

    print(qnta)
    assert()

    return answers, questions, qnta, tqna


# For the fast extraction, prepare questions x answers
def preprocessForKnowledgeFeaturesForTask2(users, posts):
    answers = posts[posts.Answer == 1].reset_index()
    questions = posts[posts.Question == 1].reset_index()
    qna = answers.set_index('ParentId').join(questions, how='inner', lsuffix='A', rsuffix='Q')
    return answers, questions, qna, qna


# Knowledge features 1: accepted_answerer_rep
def getRepOfAcceptedAnswerer(users, answers, questions, qnta, tqna):
    reputations = users.loc[:, ['Reputation']]
    # reputations = users.Reputation

    rep_accepted_ans = qnta[qnta['AcceptedAnswerIdQ'] == qnta['IdA']].set_index('OwnerUserIdA')\
        .join(reputations, how='inner').groupby('OwnerUserIdQ')['Reputation'].mean()

    # print(rep_accepted_ans)
    # assert()
    return rep_accepted_ans


# Knowledge features 1: accepted_answerer_rep
def getRepOfAcceptedAnswerer2(users, answers, questions, qnta, tqna):
    reputations = users.loc[:, ['Reputation']]
    rep_accepted_ans = qnta[qnta['AcceptedAnswerIdQ'] == qnta['IdA']]\
        .set_index('OwnerUserIdA')\
        .join(reputations, how='inner')\
        .groupby('OwnerUserIdQ')['Reputation'].mean()
    return rep_accepted_ans


# Knowledge features 2: max_rep_answerer
def getMaxRepAmongAnswerer(users, answers, questions, qnta, tqna):

    rep_max_ans = qnta.set_index('OwnerUserIdA').join(users.Reputation, how='inner')\
        .groupby('OwnerUserIdQ').Reputation.max()
    print(rep_max_ans)
    assert()
    return rep_max_ans


def getMaxRepAmongAnswerer2(users, answers, questions, qnta, tqna):
    rep_max_ans = qnta.set_index('OwnerUserIdA').join(users.Reputation, how='inner')\
        .groupby('OwnerUserIdQ').Reputation.max()
    print(rep_max_ans)
    assert()
    return rep_max_ans


# Knowledge features 3: num_que_answered
def getNumQueAnswered(users, answers, questions, qnta, tqna):
    # number of questions posted by the user that got answered
    #questions = posts[posts['PostTypeId'] == 1]
    answered_questions = questions[questions['AnswerCount'] > 0]
    return answered_questions.groupby('OwnerUserId')['AnswerCount'].count()


# Knowledge features 4: time_for_first_ans
def getTimeForFirstAns(users, answers, questions, qnta, tqna):
    tmp =  qnta[qnta['CreationDateQ'] < qnta['CreationDateA']]
    tmp['time_for_ans'] = (tmp['CreationDateA'] - tmp['CreationDateQ']).dt.total_seconds() / 60
    questions['time_for_first_ans'] = tmp.groupby(by=tmp.index)['time_for_ans'].min()
    return questions.groupby('OwnerUserId')['time_for_first_ans'].mean()


# Knowledge features 5: rep_questioner
def getAvgRepOfQuestioner(users, answers, questions, qnta, tqna):
    # Avg. reputation of the user whose question was answered
    reputations = users.loc[:, ['Reputation']]
    rep_accepted_ans = tqna.set_index('OwnerUserIdQ')\
        .join(reputations, how='inner')\
        .groupby('OwnerUserIdA')['Reputation'].mean()
    return rep_accepted_ans


# Knowledge features 6: rep_answerers
def getAvgRepOfAnswerer(users, answers, questions, qnta, tqna):
    # Avg. reputation of the users who answered the question
    reputations = users.loc[:, ['Reputation']]
    rep_accepted_ans = qnta.set_index('OwnerUserIdA')\
        .join(reputations, how='inner')\
        .groupby('OwnerUserIdQ')['Reputation'].mean()
    return rep_accepted_ans


# Knowledge features 7: rep_co_answerers
def getAvgRepOfCoAnswerer(users, answers, questions, qnta, tqna):
    reputations = users.loc[:, ['Reputation']]
    rep_ans = answers.set_index('OwnerUserId')\
        .join(reputations, how='inner')\
        .set_index('ParentId')\
        .join(questions, how='inner', lsuffix='A', rsuffix='Q')
    avg_rep_ans = rep_ans.groupby(by=rep_ans.index)['Reputation'].mean()
    rep_co_answerer = answers.set_index('ParentId')\
        .join(avg_rep_ans, how='inner')\
        .set_index('OwnerUserId')
    return rep_co_answerer.groupby(by=rep_co_answerer.index)['Reputation'].mean()


# Knowledge features 8: num_answers_recvd
def getAvgNumAnsReceived(users, answers, questions, qnta, tqna):
    #questions = posts[posts['PostTypeId'] == 1]
    return questions.fillna({'AnswerCount': 0}).groupby('OwnerUserId')['AnswerCount'].mean()




"""
# Knowledge features 1: accepted_answerer_rep
def getRepOfAcceptedAnswerer(users, posts):
    users_accepted = posts.AcceptedAnswerId.dropna().drop_duplicates().astype('int64')
    return users.loc[users_accepted, 'Reputation'].dropna()

# Knowledge features 2: max_rep_answerer
def getMaxRepAmongAnswerer(users, posts):
    return

# Knowledge features 3: num_que_answered
def getNumQueAnswered(posts):
    return

# Knowledge features 4: time_for_first_ans
def getTimeForFirstAns(posts):
    return


# Knowledge features 5: rep_questioner
def getAvgRepOfQuestioner(users, posts):
    return

# Knowledge features 6: rep_answerers
def getAvgRepOfAnswerer(users, posts):
    return



# Knowledge features 7: rep_co_answerers
def getAvgRepOfCoAnswerer(users, posts):
    return


# Knowledge features 8: num_answers_recvd
def getAvgNumAnsReceived(posts):
    return
"""
