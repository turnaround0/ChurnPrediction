import pandas as pd
from pandas.tseries.offsets import DateOffset, Day


# Dataset in Task 1
#   Posts: Extract K posts of each user
#   Users: Extract users who post at least K
def getTask1Posts(posts, K=20):
    user_k_list = posts[posts.ith == K].OwnerUserId.to_list()
    posts_k = posts[posts.OwnerUserId.isin(user_k_list)]
    return posts_k[posts_k.ith <= K]


def getTask1Users(users, posts, K):
    return users[users.numPosts >= K]


# Dataset in Task 2
#   Users: Extract users who post at least 1
#   Posts: Extract posts which create before T day from the account creation of the owner
def getTask2Posts(users, posts, T=30):
    return posts[posts.CreationDate <= (posts.CreationDateOfOwner + Day(T))]


def getTask2Users(users, posts):
    return users[users.numPosts >= 1]


# Churn in Task 1
#   Churners: Users who did not post for at least 6 months from their K-th post
#   Stayers:  Users who created at least one post within the 6 months from their K-th post
def getTask1Labels(users, posts, K):
    users_valid = posts.OwnerUserId.isin(users.index)
    posts_k = posts[(posts.ith == K) & users_valid][['CreationDate', 'OwnerUserId']]
    posts_k_next = posts[(posts.ith == K + 1) & users_valid][['CreationDate', 'OwnerUserId']]

    posts_k['Deadline'] = posts_k.CreationDate + DateOffset(months=6)
    posts_k = posts_k.drop(['CreationDate'], axis=1)
    posts_k = posts_k.merge(posts_k_next, how='left', on='OwnerUserId')
    posts_k.CreationDate = posts_k.CreationDate.fillna(pd.to_datetime('2100-12-31'))

    users_k = users[users.numPosts >= K]
    users_k = users_k.drop(users_k.columns, axis=1)
    users_k['is_churn'] = 0
    # If creation date of K + 1 post is in deadline, the user is a stayer.
    users_k.loc[posts_k[posts_k.CreationDate > posts_k.Deadline].OwnerUserId, 'is_churn'] = 1
    return users_k


# Churn in Task2
#   Churners: Users who did not post for at least 6 months from T days after account creation
#   Stayers:  Users who created at least one post within the 6 months from T days after account creation
def getTask2Labels(users, posts, T=30):
    users = users[users.numPosts > 0]
    posts = posts[posts.OwnerUserId.isin(users.index)]
    deadline_t = posts.CreationDateOfOwner + Day(T)
    deadline_churn = deadline_t + DateOffset(months=6)

    posts_t = posts[posts.CreationDate <= deadline_t]
    posts_t_users = posts_t.OwnerUserId.unique()
    posts_after_t = posts[(posts.CreationDate <= deadline_churn) & (posts.CreationDate > deadline_t) &
                          (posts.OwnerUserId.isin(posts_t_users))]

    users_t = users.loc[posts_t_users]
    users_t = users_t.drop(users_t.columns, axis=1)
    users_t['is_churn'] = 1
    # If there are posts after T in deadline, the user is a stayer.
    users_t.loc[posts_after_t.OwnerUserId, 'is_churn'] = 0
    return users_t
