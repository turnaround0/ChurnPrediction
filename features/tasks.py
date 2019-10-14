import pandas as pd


# Dataset in Task 1
#   Posts: Extract K posts of each user
def getTask1Posts(posts, K=20):
    posts_k = posts.groupby('OwnerUserId').nth(K)
    return posts_k


#   Users: Extract users who post at least K
def getTask1Users(users, posts, K=20):
    kth_posts = getTask1Posts(posts, K)
    users_k = users[users.index.isin(kth_posts.index)]
    return users_k


# Dataset in Task 2
#   Users: Extract users who post at least 1
#   Posts: Extract posts which create before T day from the account creation of the owner
def getTask2Posts(users, posts, T=30):
    users = getTask1Users(users, posts, K=1)
    posts = posts[posts.OwnerUserId.isin(users.index)]
    posts_t = posts[(posts.CreationDate - posts.CreationDateOfOwner).dt.days <= T]
    return posts_t


# Churn in Task 1
#   Churners: Users who did not post for at least 6 months from their K-th post
#   Stayers:  Users who created at least one post within the 6 months from their K-th post
def getTask1Labels(users, posts, K=20):
    label_df = users.drop(users.columns, axis=1)
    label_df = getTask1Users(label_df, posts, K=K)

    posts_k = getTask1Posts(posts, K=K)
    posts_k_next = getTask1Posts(posts, K=K+1)
    diff = posts_k_next.loc[posts_k.index, 'CreationDate'] - posts_k.CreationDate
    diff = diff.fillna(pd.Timedelta(days=181))

    label_df['is_churn'] = 0.0
    label_df[diff.loc[label_df.index] > pd.Timedelta(days=180)] = 1.0
    return label_df


# Churn in Task2
#   Churners: Users who did not post for at least 6 months from T days after account creation
#   Stayers:  Users who created at least one post within the 6 months from T days after account creation
def getTask2Labels(users, posts, T=30):
    label_df = users.drop(users.columns, axis=1)
    label_df = getTask1Users(label_df, posts, K=1)

    posts_t = getTask2Posts(users, posts, T=T)
    num_posts_t = posts_t.groupby('OwnerUserId').size()

    posts_t_180 = getTask2Posts(users, posts, T=T+180)
    num_posts_t_180 = posts_t_180.groupby('OwnerUserId').size()

    num_new_posts = (num_posts_t_180 - num_posts_t).fillna(0)
    num_new_posts = num_new_posts[num_new_posts > 0]

    label_df['is_churn'] = 1.0
    label_df.loc[num_new_posts.index] = 0.0
    return label_df
