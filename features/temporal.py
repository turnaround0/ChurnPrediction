import pandas as pd


# Temporal features 1: gap1
def getTimeGap1OfUser(users, posts):
    # CreationDateOfOwner of users is merged to posts dataframe when preprocessing data
    posts_group_1 = posts.groupby('OwnerUserId').nth(1)
    gap_1 = (posts_group_1.CreationDate - posts_group_1.CreationDateOfOwner).dropna() / pd.Timedelta('1 minute')
    gap_1.index.name = 'Id'
    return gap_1


# Temporal features 2: gapK
def getTimeGapsOfPosts(posts, K):
    posts_group = posts.groupby('OwnerUserId')
    posts_group_k_prior = posts_group.nth(K - 1)
    posts_group_k = posts_group.nth(K)
    gap_k = (posts_group_k.CreationDate - posts_group_k_prior.CreationDate).dropna() / pd.Timedelta('1 minute')
    gap_k.index.name = 'Id'
    return gap_k


# Temporal features 3: last_gap
def getTimeLastGapOfPosts(posts):
    gap_last = getTimeGapsOfPosts(posts, -1)
    gap_last.index.name = 'Id'
    return gap_last


# Temporal features 4: time_since_last_post
def getTimeSinceLastPost(users, posts):
    end_date = pd.to_datetime('2012-07-31')
    posts_group_last = posts.groupby('OwnerUserId').nth(-1)
    gap_since_last = pd.to_datetime(end_date) - posts_group_last.CreationDate
    gap_since_last.index.name = 'Id'
    return gap_since_last


# Temporal features 5: mean_gap
def getTimeMeanGap(posts):
    posts_group = posts.groupby('OwnerUserId')
    assert()
    # gap_posts = posts[['gap', 'OwnerUserId']]
    # gap_posts = gap_posts[gap_posts.gap != 0]
    # return gap_posts.groupby('OwnerUserId').mean()
    return posts
