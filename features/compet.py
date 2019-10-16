# Competitiveness features 1: relative_rank_pos
def getRelRankPos(posts):
    answers = posts[posts.PostTypeId == 2]
    answers['ans_score_div_ranks'] = answers.Score / answers.groupby('ParentId').Score.rank(ascending=False)
    return answers.groupby('OwnerUserId').ans_score_div_ranks.mean()
