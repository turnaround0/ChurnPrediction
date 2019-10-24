# Answering features 1: Average of Answer Count
def getAvgNumOfAnswerCount(users, answers, questions, qnta, tqna):
    return answers.groupby('OwnerUserId').AnswerCount.sum()


# Answering features 1: First Type of Posting
def getFirstPostType(posts):
    tmp = posts[posts.ith == 1].OwnerUserId.to_frame()
    tmp = tmp.set_index('OwnerUserId')
    return posts[posts.OwnerUserId.isin(tmp.index)].PostTypeId.apply(lambda post_type: 1 if post_type == 2 else 0)


# Answering features 1: Total # of comment
def getTotalNumOfComments(users, answers, questions,  qnta, tqna):
    tqna['total_comment'] = tqna['CommentCountA'] + tqna['CommentCountQ'] / tqna['AnswerCountQ']
    return tqna.groupby('OwnerUserIdQ').total_comment.sum()
