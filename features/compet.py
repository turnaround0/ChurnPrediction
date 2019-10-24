# Competitiveness features 1: relative_rank_pos
def getRelRankPos(posts):
    # average of total # of answers for a question divided by the rank of user's answer
    answers = posts[posts.PostTypeId == 2]
    group_score = answers.groupby('ParentId').Score
    answers['rel_rank_pos'] = group_score .transform('count') / group_score.rank(method='average')
    return answers.groupby('OwnerUserId').rel_rank_pos.mean()
