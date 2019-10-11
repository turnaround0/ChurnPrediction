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
