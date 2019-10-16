# Content features 1: ans_length
def getLengthOfAnswers(posts):
    return posts[posts.PostTypeId == 2].groupby('OwnerUserId').BodyLen.mean()


# Content features 2: que_length
def getLengthOfQuestions(posts):
    return posts[posts.PostTypeId == 1].groupby('OwnerUserId').BodyLen.mean()
