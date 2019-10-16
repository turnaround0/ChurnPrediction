# Quality features 1: ans_score
def getScoreOfAnswers(posts):
    return posts[posts.PostTypeId == 2].groupby('OwnerUserId').Score.mean()


# Quality features 2: que_score
def getScoreOfQuestions(posts):
    return posts[posts.PostTypeId == 1].groupby('OwnerUserId').Score.mean()
