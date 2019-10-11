# Frequency features 1: num_answers
def getNumAnswers(posts_group):
    return posts_group.Answer.sum()


# Frequency features 2: num_questions
def getNumQuestions(posts_group):
    return posts_group.Question.sum()


# Frequency features 3: ans_ques_ratio
def getAnsQuesRatio(num_answers, num_questions):
    df = num_answers / num_questions
    df[num_questions == 0] = 0
    return df


# Frequency features 4: num_posts
def getNumPosts(posts_group):
    return posts_group.size()
