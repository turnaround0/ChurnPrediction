import pandas as pd


# Temporal features 1: gap1
def getTimeGap1OfUser(users, posts):
    # CreationDateOfOwner of users is merged to posts dataframe when preprocessing data
    posts_first = posts.groupby('OwnerUserId').first()
    return (posts_first.CreationDate - posts_first.CreationDateOfOwner).dt.total_seconds() / 60


# Temporal features 2: gapK
def getTimeGapkOfPosts(posts, k):
    posts_k_minus_1_dates = posts[posts.ith == k - 1].set_index('OwnerUserId').CreationDate
    posts_k_dates = posts[posts.ith == k].set_index('OwnerUserId').CreationDate
    return (posts_k_dates - posts_k_minus_1_dates).dt.total_seconds() / 60


# Temporal features 3: last_gap
def getTimeLastGapOfPosts(posts):
    last_dates = posts.groupby('OwnerUserId').CreationDate.nth(-1)
    before_last_dates = posts.groupby('OwnerUserId').CreationDate.nth(-2)
    return (last_dates - before_last_dates).dt.total_seconds() / 60


# Temporal features 4: time_since_last_post
def getTimeSinceLastPost(users, posts, T):
    last_dates = posts.groupby('OwnerUserId').CreationDate.last()
    deadline_t = users.CreationDate + pd.offsets.Day(T)
    return (deadline_t - last_dates).dt.total_seconds() / 60


# Temporal features 5: mean_gap
def getTimeMeanGap(posts):
    last_dates = posts.groupby('OwnerUserId').CreationDate.last()
    first_dates = posts.groupby('OwnerUserId').CreationDate.first()
    num_posts = posts.groupby('OwnerUserId').size()
    return (last_dates - first_dates).dt.total_seconds() / 60 / num_posts
