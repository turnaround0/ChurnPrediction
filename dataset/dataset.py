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
    df['Question'] = 0
    df.loc[df.PostTypeId == 1, 'Question'] = 1
    df['Answer'] = 0
    df.loc[df.PostTypeId == 2, 'Answer'] = 1

    df = df.merge(user_df['CreationDate'].rename('CreationDateOfOwner'),
                  how='left', left_on='OwnerUserId', right_on='Id')

    df = df.set_index('Id')
    return df.drop(['PostTypeId', 'Body'], axis=1)


def set_posts_ith(posts):
    posts = posts.sort_values(by=['OwnerUserId', 'CreationDate']).reset_index()
    posts['ithRow'] = posts.index
    first_posts = posts.groupby('OwnerUserId').first()
    tmp_posts = posts.merge(first_posts, on='OwnerUserId')
    posts['ith'] = tmp_posts.ithRow_x - tmp_posts.ithRow_y + 1
    return posts.drop(['ithRow'], axis=1)


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


def load_data(dataset_type):
    data_paths = {
        'tiny': 'dataset/tiny/',
        'small': 'dataset/small/',
        'full': 'dataset/full/'
    }
    data_path = data_paths[dataset_type]

    dataset_names = ['Users', 'Posts']
    df_list = []
    for dataset_name in dataset_names:
        pkl_file_path = data_path + dataset_name + '.pkl'

        if os.path.exists(pkl_file_path):
            df_list.append(load_from_pkl(pkl_file_path))
        else:
            xml_file_path = data_path + dataset_name + '.xml'
            df = xml2df(xml_file_path)

            # Cut dataset by given period
            # In case user, it should be ended 6 months earlier due to check churn.
            start_time = pd.to_datetime('2008-07-31')
            if dataset_name == 'Posts':
                end_time = pd.to_datetime('2012-07-31')
            else:
                if dataset_type == 'tiny':  # tiny dataset doesn't have matched user data
                    end_time = pd.to_datetime('2012-07-31')
                else:
                    end_time = pd.to_datetime('2012-01-31')
            df.CreationDate = pd.to_datetime(df.CreationDate)
            df = df[(df.CreationDate >= start_time) & (df.CreationDate <= end_time)]

            if dataset_name == 'Posts':
                df = posts_preprocess(df, df_list[0])
                df = set_posts_ith(df)
            else:
                df = users_preprocess(df)

            save_to_pkl(df, pkl_file_path)
            df_list.append(df)

    return df_list[0], df_list[1]
