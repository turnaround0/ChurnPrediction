# Consistency features 1: ans_stddev
def getStdevOfScoresOfAnswers(posts):
    return posts[posts.PostTypeId == 2].groupby('OwnerUserId').Score.std()


# Consistency features 2: que_stddev
def getStdevOfScoresOfQuestions(posts):
    return posts[posts.PostTypeId == 1].groupby('OwnerUserId').Score.std()
