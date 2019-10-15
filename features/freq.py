# Frequency features 1: num_answers
def getNumAnswers(posts):
    return posts[posts.PostTypeId == 2].groupby('OwnerUserId').size()


# Frequency features 2: num_questions
def getNumQuestions(posts):
    return posts[posts.PostTypeId == 1].groupby('OwnerUserId').size()


# Frequency features 3: ans_ques_ratio
def getAnsQuesRatio(num_answers, num_questions):
    # Use Laplace Smoothing
    return (num_answers + 1) / (num_questions + 1)


# Frequency features 4: num_posts
def getNumPosts(posts):
    return posts.groupby('OwnerUserId').size()
