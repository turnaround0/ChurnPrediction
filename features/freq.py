# Frequency features 1: num_answers
def getNumAnswers(posts_group):
    return posts_group['AnswerCount'].sum()


# Frequency features 2: num_questions
def getNumQuestions(posts_group):
    return posts_group.size()


# Frequency features 3: ans_ques_ratio
def getAnsQuesRatio(num_answers, num_questions):
    return num_answers / num_questions


# Frequency features 4: num_posts
def getNumPosts(posts_group):
    return posts_group.size() + posts_group['AnswerCount'].sum()
