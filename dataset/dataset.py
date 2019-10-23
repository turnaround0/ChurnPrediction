import time
import pandas as pd
import xmltodict
import os.path


def users_preprocess(df):
    # Drop unused columns and change column types
    df = df[['Id', 'Reputation', 'CreationDate', 'LastAccessDate']]

    df.Id = df.Id.astype('int64')
    df.Reputation = df.Reputation.astype('int64')
    df.LastAccessDate = pd.to_datetime(df.LastAccessDate)

    df = df.set_index('Id')
    return df.drop([-1])


def posts_preprocess(df, user_df):
    # Drop unused columns
    df = df[['Id', 'PostTypeId', 'CreationDate', 'AcceptedAnswerId', 'ParentId',
             'Score', 'OwnerUserId', 'AnswerCount', 'CommentCount', 'Body']]
    df['BodyLen'] = df.Body.str.len()

    df.Id = df.Id.astype('int64')
    df = df.dropna(subset=['OwnerUserId'])
    df.OwnerUserId = df.OwnerUserId.astype('int64')
    df.PostTypeId = df.PostTypeId.astype('int64')
    df.AnswerCount = df.AnswerCount.fillna(0).astype('int64')
    df.CommentCount = df.CommentCount.fillna(0).astype('int64')
    df.AcceptedAnswerId = df.AcceptedAnswerId.fillna(0).astype('int64')
    df.ParentId = df.ParentId.fillna(0).astype('int64')
    df.Score = df.Score.astype('int64')
    df.PostTypeId = df.PostTypeId.astype('int64')

    df = df.merge(user_df.CreationDate.rename('CreationDateOfOwner'),
                  how='left', left_on='OwnerUserId', right_on='Id')

    df = df.set_index('Id')
    return df.drop(['Body'], axis=1)


def full_posts_preprocess(df):
    df.AnswerCount = df.AnswerCount.fillna(0).astype('int64')
    df.CommentCount = df.CommentCount.fillna(0).astype('int64')
    df.AcceptedAnswerId = df.AcceptedAnswerId.fillna(0).astype('int64')
    df.ParentId = df.ParentId.fillna(0).astype('int64')
    return df.rename(columns={'BodyWordNum': 'BodyLen'})


def set_posts_ith(posts):
    posts = posts.sort_values(by=['OwnerUserId', 'CreationDate']).reset_index()
    posts['ithRow'] = posts.index
    first_posts = posts.groupby('OwnerUserId').first()
    tmp_posts = posts.merge(first_posts, on='OwnerUserId')
    posts['ith'] = tmp_posts.ithRow_x - tmp_posts.ithRow_y + 1
    return posts.drop(['ithRow'], axis=1).set_index('Id', drop=True)


# Save and Load dataframe
def save_to_pkl(df, pkl_file_path):
    df.to_pickle(pkl_file_path)


def load_from_pkl(pkl_file_path):
    return pd.read_pickle(pkl_file_path)


def xml2df(xml_path):
    # Read xml file and transform to pandas dataframe
    with open(xml_path, 'r', encoding='UTF8') as f:
        data = f.read()
        xml_dict = xmltodict.parse(data)
        key = list(xml_dict.keys())[0]

        df = pd.DataFrame(xml_dict[key]['row'])
        df.columns = [col.replace('@', '') for col in df.columns]

    return df


def cut_posts_by_period(dataset_name, df):
    # You should extract the dataset for the period of the dataset: July31,2008 ~ July31,2012
    # In case Users, it should be ended 6 months earlier due to check churn.
    start_time = pd.to_datetime('2008-07-31')
    end_time = pd.to_datetime('2012-07-31') if dataset_name == 'Posts' else pd.to_datetime('2012-01-31')
    df = df[(df.CreationDate >= start_time) & (df.CreationDate <= end_time)]
    return df


def load_dataset(dataset_type):
    # Link: https://archive.org/details/stackexchange
    data_paths = {
        'tiny': 'dataset/tiny/',    # academia.meta.stackexchange.com.7z
        'small': 'dataset/small/',  # math.stackexchange.com.7z
        'full': 'dataset/full/'     # stackoverflow.com-Users.7z, stackoverflow.com-Posts.7z
    }
    data_path = data_paths[dataset_type]
    dataset_names = ['Users', 'Posts']
    df_list = []

    print('*** Loading dataset ***')
    start_time = time.time()

    for dataset_name in dataset_names:
        pkl_file_path = data_path + dataset_name + '.pkl'
        pkl_file_reduce_path = data_path + dataset_name + '_reduce.pkl'

        if os.path.exists(pkl_file_path):
            df = load_from_pkl(pkl_file_path)
        elif dataset_type == 'full' and os.path.exists(pkl_file_reduce_path):
            # In case of full dataset, reduced pickle files were given.
            # Link: https://drive.google.com/drive/folders/1Fp_7GDH_t7xfnU8aXeKrcBC54_nECOcu
            df = load_from_pkl(pkl_file_reduce_path)
            df = cut_posts_by_period(dataset_name, df)

            if dataset_name == 'Posts':
                df = full_posts_preprocess(df)
                df = set_posts_ith(df)
            else:
                df = df.drop([-1])

            save_to_pkl(df, pkl_file_path)
        else:
            xml_file_path = data_path + dataset_name + '.xml'
            df = xml2df(xml_file_path)
            df.CreationDate = pd.to_datetime(df.CreationDate)
            df = cut_posts_by_period(dataset_name, df)

            if dataset_name == 'Posts':
                df = posts_preprocess(df, df_list[0])
                df = set_posts_ith(df)
            else:
                df = users_preprocess(df)

            save_to_pkl(df, pkl_file_path)

        df_list.append(df)
        print(dataset_name, 'shape:', df_list[-1].shape)

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')

    return df_list[0], df_list[1]


def store_features(list_of_K, list_of_T, features_of_task1, features_of_task2, file_type='csv'):
    print('*** Store features ***')
    print('File type:', file_type)
    start_time = time.time()

    for K in list_of_K:
        if file_type == 'csv':
            features_of_task1[K].to_csv('output/features/task1_{}posts_features.csv'.format(K))
        else:
            features_of_task1[K].to_pickle('output/features/task1_{}posts_features.pkl'.format(K))

    for T in list_of_T:
        if file_type == 'csv':
            features_of_task2[T].to_csv('output/features/task2_{}days_features.csv'.format(T))
        else:
            features_of_task2[T].to_pickle('output/features/task2_{}days_features.pkl'.format(T))

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')


def restore_features(list_of_K, list_of_T, file_type='csv'):
    print('*** Restore features ***')
    print('File type:', file_type)
    start_time = time.time()

    features_of_task1, features_of_task2 = {}, {}

    for K in list_of_K:
        if file_type == 'csv':
            features_of_task1[K] = pd.read_csv('output/features/task1_{}posts_features.csv'.format(K))
        else:
            features_of_task1[K] = pd.read_pickle('output/features/task1_{}posts_features.pkl'.format(K))

    for T in list_of_T:
        if file_type == 'csv':
            features_of_task2[T] = pd.read_csv('output/features/task2_{}days_features.csv'.format(T))
        else:
            features_of_task2[T] = pd.read_pickle('output/features/task2_{}days_features.pkl'.format(T))

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
    return features_of_task1, features_of_task2


def preprocess(users, posts):
    print('*** Preprocess ***')
    start_time = time.time()

    users['numPosts'] = posts.groupby('OwnerUserId').size()
    posts = posts[posts.OwnerUserId.isin(users.index)]

    end_time = time.time()
    print('Processing time:', round(end_time - start_time, 8), 's')
    return users, posts


def _print_stats(list_of_items, features):
    data_list = []
    for item in list_of_items:
        count = features[item].groupby('is_churn').size()
        percentage = round(count[0] / (count[0] + count[1]) * 100, 4)
        ratio = str(round(count[0] / count[1], 4)) + ':1'

        data_list.append((item, count[1], count[0], percentage, str(ratio)))
        print('#', item, 'Churn:', count[1], 'Stay:', count[0], 'Percentage:', percentage, 'Ratio:', ratio)

    df = pd.DataFrame(data_list)
    df.columns = ['Id', 'Churner', 'Stayer', 'Stayer percentage', 'Ratio (Stayer:Churner)']
    return df.set_index('Id')


def print_stats(list_of_K, list_of_T, features_of_task1, features_of_task2):
    print('\nDataset characteristic by K')
    df_k = _print_stats(list_of_K, features_of_task1).rename_axis('K')
    df_k.to_csv('output/dataset_char_K.csv')

    print('\nDataset characteristic by T')
    df_t = _print_stats(list_of_T, features_of_task2).rename_axis('T')
    df_t.to_csv('output/dataset_char_T.csv')
