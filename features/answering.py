# For the fast extraction, prepare questions x answers
def prepareTask1(posts, all_posts):
    answers = posts[posts.PostTypeId == 2][['OwnerUserId', 'ParentId', 'CommentCount']]
    answers.columns = ['AnswerUserId', 'QuestionId', 'CommentCountA']
    all_answers = all_posts[all_posts.PostTypeId == 2][['OwnerUserId', 'ParentId', 'CommentCount']]
    all_answers.columns = ['AnswerUserId', 'QuestionId', 'CommentCountA']

    questions = posts[posts.PostTypeId == 1][['OwnerUserId', 'AnswerCount', 'CommentCount']]\
        .rename(columns={'OwnerUserId': 'QuestionUserId', 'CommentCount': 'CommentCountQ'})
    all_questions = all_posts[all_posts.PostTypeId == 1][['OwnerUserId', 'AnswerCount', 'CommentCount']]\
        .rename(columns={'OwnerUserId': 'QuestionUserId', 'CommentCount': 'CommentCountQ'})

    # Questions and Total Answers
    qnta = all_answers.merge(questions, left_on='QuestionId', right_index=True)
    # Total Questions and Answers
    tqna = answers.merge(all_questions, left_on='QuestionId', right_index=True)

    return answers, questions, qnta, tqna


# For the fast extraction, prepare questions x answers
def prepareTask2(posts):
    answers = posts[posts.PostTypeId == 2][['OwnerUserId', 'ParentId', 'CommentCount']]
    answers.columns = ['AnswerUserId', 'QuestionId', 'CommentCountA']
    questions = posts[posts.PostTypeId == 1][['OwnerUserId', 'AnswerCount', 'CommentCount']]\
        .rename(columns={'OwnerUserId': 'QuestionUserId', 'CommentCount': 'CommentCountQ'})

    ans_ques = answers.merge(questions, left_on='QuestionId', right_index=True)

    return answers, questions, ans_ques, ans_ques


# Answering features 1: Average of Answer Count
def getAvgNumOfAnswerCount(tqna):
    return tqna.groupby('AnswerUserId').AnswerCount.sum()


# Answering features 1: First Type of Posting
def getFirstPostTypeIsAnswer(posts):
    tmp = posts[posts.ith == 1].OwnerUserId.to_frame()
    tmp = tmp.set_index('OwnerUserId')
    return posts[posts.OwnerUserId.isin(tmp.index)].PostTypeId.apply(lambda post_type: 1 if post_type == 2 else 0)


# Answering features 1: Total # of comment
def getTotalNumOfComments(tqna):
    tqna['total_comment'] = tqna.CommentCountA + tqna.CommentCountQ / tqna.AnswerCount
    return tqna.groupby('QuestionUserId').total_comment.sum()
