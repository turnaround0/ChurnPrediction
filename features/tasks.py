import pandas as pd


# Dataset in Task 1
#   Posts: Extract K posts of each user
#   Users: Extract users who post at least K
def getTask1Posts(posts_group, K=20):
    kth_group_posts = posts_group.nth(K)
    return kth_group_posts


def getTask1Users(users, posts_group, K=20):
    kth_posts = getTask1Posts(posts_group, K)
    return users[users.index.isin(kth_posts.index)]


# Dataset in Task 2
#   Users: Extract users who post at least 1
#   Posts: Extract posts which create before T day from the account creation of the owner
def getTask2Posts(users, posts, posts_group, T=30):
    users = getTask1Users(users, posts_group, 1)
    posts = posts[posts.OwnerUserId.isin(users.index)]
    return posts[(posts.CreationDate - posts.CreationDateOfOwner).dt.days <= T]


# Churn in Task 1
#   Churners: Users who did not post for at least 6 months from their K-th post
#   Stayers:  Users who created at least one post within the 6 months from their K-th post
def getTask1Labels(users, posts_group, K=20):
    label_df = getTask1Users(users, posts_group, K=K)
    label_df = label_df.drop(label_df.columns, axis=1)

    posts_k = getTask1Posts(posts_group, K=K)
    posts_k_plus_1 = getTask1Posts(posts_group, K=K+1)
    diff = posts_k_plus_1.loc[posts_k.index, 'CreationDate'] - posts_k.CreationDate
    diff = diff.fillna(pd.Timedelta(days=181))

    label_df['is_churn'] = 0.0
    label_df[diff.loc[label_df.index] > pd.Timedelta(days=180)] = 1.0
    return label_df


# Churn in Task2
#   Churners: Users who did not post for at least 6 months from T days after account creation
#   Stayers:  Users who created at least one post within the 6 months from T days after account creation
def getTask2Labels(users, posts, posts_group, T=30):
    label_df = getTask1Users(users, posts_group, K=1)
    label_df = label_df.drop(label_df.columns, axis=1)

    posts_t = getTask2Posts(users, posts, posts_group, T=T)
    num_posts_t = posts_t.groupby('OwnerUserId').size()

    posts_t_180 = getTask2Posts(users, posts, posts_group, T=T+180)
    num_posts_t_180 = posts_t_180.groupby('OwnerUserId').size()

    num_new_posts = (num_posts_t_180 - num_posts_t).fillna(0)
    num_new_posts = num_new_posts[num_new_posts > 0]

    label_df['is_churn'] = 1.0
    label_df.loc[num_new_posts.index.tolist()] = 0.0

    return label_df
