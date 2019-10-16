# Gratitude features 1: ans_comments
def getAvgNumOfAnswers(posts):
    return posts[posts.PostTypeId == 2].groupby('OwnerUserId').CommentCount.mean()


# Gratitude features 2: que_comments
def getAvgNumOfQuestions(posts):
    return posts[posts.PostTypeId == 1].groupby('OwnerUserId').CommentCount.mean()
