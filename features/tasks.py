import pandas as pd


# Dataset in Task 1
#   Posts: Extract K posts of each user
def getTask1Posts(posts, K=20):
    user_k_list = posts[posts.ith == K].OwnerUserId.to_list()
    posts_k = posts[posts.OwnerUserId.isin(user_k_list)]
    return posts_k[posts_k.ith <= K]


#   Users: Extract users who post at least K
def getTask1Users(users, posts, K):
    return users[users.numPosts >= K]


# Dataset in Task 2
#   Users: Extract users who post at least 1
#   Posts: Extract posts which create before T day from the account creation of the owner
def getTask2Posts(users, posts, T=30):
    return posts[(posts.CreationDate - posts.CreationDateOfOwner).dt.days <= T]


def getTask2Users(users, posts):
    return users[users.numPosts >= 1]


# Churn in Task 1
#   Churners: Users who did not post for at least 6 months from their K-th post
#   Stayers:  Users who created at least one post within the 6 months from their K-th post
def prepareFeaturesTask1(users, posts, K):
    tmp = posts[posts['ith'] == K]['OwnerUserId'].to_frame()
    tmp = tmp.set_index('OwnerUserId')
    posts = posts[posts['OwnerUserId'].isin(tmp.index)]

    posts_task = posts[posts['OwnerUserId'].isin(users.index)]
    posts_Kth_time = posts_task[posts_task['ith'] == K]
    posts_Kth_time = posts_Kth_time.set_index('OwnerUserId')['CreationDate']
    posts_deadline = posts_Kth_time + pd.tseries.offsets.DateOffset(months=6)

    posts_stayer = posts_task[posts_task['ith'] > K].groupby('OwnerUserId')['CreationDate'].min().to_frame()
    posts_stayer = posts_stayer.merge(posts_deadline, on='OwnerUserId', how='left', suffixes=('_left', '_right'))

    posts_churner1 = posts_stayer[posts_stayer['CreationDate_left'] > posts_stayer['CreationDate_right']]
    posts_churner1['is_churn'] = 1
    posts_churner1 = posts_churner1[['is_churn']]
    posts_stayer = posts_stayer[posts_stayer['CreationDate_left'] <= posts_stayer['CreationDate_right']]
    posts_stayer['is_churn'] = 0
    posts_stayer = posts_stayer[['is_churn']]

    posts_churner2 = posts_task[posts_task['ith'] >= K].groupby('OwnerUserId').count()
    posts_churner2 = posts_churner2[posts_churner2['CreationDate'] == 1][['CreationDate']]
    posts_churner2['is_churn'] = 1
    posts_churner2 = posts_churner2[['is_churn']]

    posts = pd.concat([posts_stayer, posts_churner1, posts_churner2])
    posts = posts.rename(columns={'OwnerUserId': 'Id'})
    users['is_churn'] = 0
    users.update(posts)
    return users


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


# Churn in Task2
#   Churners: Users who did not post for at least 6 months from T days after account creation
#   Stayers:  Users who created at least one post within the 6 months from T days after account creation
def prepareFeaturesTask2(users, posts, T=30):
    users = getTask1Users(users, posts, K=1)
    observe_deadline = posts['CreationDateOfOwner'] + pd.offsets.Day(T)
    churn_deadline = observe_deadline + pd.tseries.offsets.DateOffset(months=6)
    posts_observed = posts[(posts['CreationDate'] <= observe_deadline) & (posts['CreationDate'] >= posts['CreationDateOfOwner'])]
    posts_after_observe = posts[(posts['CreationDate'] <= churn_deadline) & (posts['CreationDate'] > observe_deadline)]
    label_df = users.reindex((posts_observed.groupby('OwnerUserId')['OwnerUserId'].count() > 0).index)
    stayers = (posts_after_observe.groupby('OwnerUserId')['OwnerUserId'].count() > 0).index
    churners = list(set(label_df.index) - set(stayers))
    label_df['is_churn'] = 0.
    label_df.loc[churners, 'is_churn'] = 1.
    return label_df
