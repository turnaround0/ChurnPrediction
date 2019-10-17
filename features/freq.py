# Frequency features 1: num_answers
def getNumAnswers(posts):
    return posts[posts.PostTypeId == 2].groupby('OwnerUserId').size()


# Frequency features 2: num_questions
def getNumQuestions(posts):
    return posts[posts.PostTypeId == 1].groupby('OwnerUserId').size()


# Frequency features 3: ans_ques_ratio
def getAnsQuesRatio(num_answers, num_questions):
    # Don't use Laplace Smoothing
    # If #ans is 0, return 0. If #ques is 0, return NaN
    return num_answers / num_questions


# Frequency features 4: num_posts
def getNumPosts(posts):
    return posts.groupby('OwnerUserId').size()
