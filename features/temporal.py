import pandas as pd


# Temporal features 1: gap1
def getTimeGap1OfUser(posts_group):
    posts_group_1 = posts_group.nth(1)
    gap1 = posts_group_1.CreationDate - posts_group_1.CreationDateOfOwner
    return gap1


# Temporal features 2: gapK
def getTimeGapsOfPosts(posts_group, K):
    posts_group_k_minus_1 = posts_group.nth(K - 1)
    posts_group_k = posts_group.nth(K)

    gap_k = (posts_group_k.CreationDate - posts_group_k_minus_1.CreationDate).dropna()
    return gap_k


# Temporal features 3: last_gap
def getTimeLastGapOfPosts(posts_group):
    return getTimeGapsOfPosts(posts_group, -1)


# Temporal features 4: time_since_last_post
def getTimeSinceLastPost(posts_group, end_date):
    posts_group_last = posts_group.nth(-1)
    gap_since_last = pd.to_datetime(end_date) - posts_group_last.CreationDate
    return gap_since_last


# Temporal features 5: mean_gap
def getTimeMeanGap(posts):
    gap_posts = posts[['gap', 'OwnerUserId']]
    gap_posts = gap_posts[gap_posts.gap != 0]
    return gap_posts.groupby('OwnerUserId').mean()
