import pandas as pd
import xmltodict
import os.path


def users_light_preprocess(df):
    # Drop unused columns
    return df[['Id', 'Reputation', 'CreationDate', 'LastAccessDate']]


def users_heavy_preprocess(df):
    # Drop unused columns and change column types
    df = df[['Id', 'Reputation', 'CreationDate', 'LastAccessDate']]

    df.LastAccessDate = pd.to_datetime(df.LastAccessDate)
    df.Id = df.Id.astype('int64')
    df.Reputation = df.Reputation.astype('int64')

    df = df.set_index('Id')
    df = df.drop([-1])
    return df


def get_gap(df):
    df = df[['OwnerUserId', 'CreationDate']]
    df['gap'] = 0
    df_group = df.groupby('OwnerUserId')

    for key in df_group.groups:
        prev_date = pd.Timedelta(0)
        for group in df_group.groups[key]:
            if prev_date == pd.Timedelta(0):
                prev_date = df.loc[group].CreationDate
            else:
                gap = df.loc[group].CreationDate - prev_date
                df.loc[group, 'gap'] = gap / pd.Timedelta('1 minute')
                prev_date = df.loc[group].CreationDate
    return df.gap


def posts_light_preprocess(df):
    # Drop unused columns
    df = df[['Id', 'PostTypeId', 'CreationDate', 'AcceptedAnswerId', 'ParentId',
             'Score', 'OwnerUserId', 'AnswerCount', 'CommentCount', 'Body']]
    df['BodyLen'] = df.Body.str.len()
    return df.drop(['Body'], axis=1)


def posts_heavy_preprocess(df, user_df):
    df.Id = df.Id.astype('int64')
    df = df.dropna(subset=['OwnerUserId'])
    df.OwnerUserId = df.OwnerUserId.astype('int64')
    df.PostTypeId = df.PostTypeId.astype('int64')
    df.AnswerCount = df.AnswerCount.fillna(0).astype('int64')
    df.CommentCount = df.CommentCount.fillna(0).astype('int64')

    df = df.set_index('Id')
    user_s = user_df['CreationDate']
    df = df.merge(user_s.rename('CreationDateOfOwner'), how='left', left_on='OwnerUserId', right_on='Id')

    df['gap'] = get_gap(df)
    df['Question'] = 0
    df.loc[df.PostTypeId == 1, 'Question'] = 1
    df['Answer'] = 0
    df.loc[df.PostTypeId == 2, 'Answer'] = 1

    df = df.drop(['PostTypeId'], axis=1)
    return df


# Save and Load dataframe
def save_to_pkl(df, pkl_file_path):
    df.to_pickle(pkl_file_path)


def load_from_pkl(pkl_file_path):
    return pd.read_pickle(pkl_file_path)


def xml2df(xml_path):
    # Read xml file and transform to pandas dataframe
    with open(xml_path, 'r', encoding='UTF8') as f:
        data = ''
        while True:
            c = f.read(1024 * 1024 * 1024)
            if not c:
                break
            else:
                data += c

        #data = f.read()
        xml_dict = xmltodict.parse(data)
        key = list(xml_dict.keys())[0]

        df = pd.DataFrame(xml_dict[key]['row'])
        df.columns = [col.replace('@', '') for col in df.columns]

    return df


def load_data(dataset_type, start_time, end_time):
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
        pure_pkl_file_path = data_path + dataset_name + '_pure.pkl'

        if os.path.exists(pkl_file_path):
            df_list.append(load_from_pkl(pkl_file_path))
        elif os.path.exists(pure_pkl_file_path):
            df = load_from_pkl(pure_pkl_file_path)
            print(df)

            if dataset_name == 'Posts':
                df = posts_heavy_preprocess(df, df_list[0])
            else:
                df = users_heavy_preprocess(df)

            # save_to_pkl(df, pkl_file_path)
            df_list.append(df)
        else:
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            xml_file_path = data_path + dataset_name + '.xml'

            df = xml2df(xml_file_path)

            # Cut dataset by given period
            df.CreationDate = pd.to_datetime(df.CreationDate)
            df = df[(df.CreationDate >= start_time) & (df.CreationDate <= end_time)]

            if dataset_name == 'Posts':
                df = posts_light_preprocess(df)
                save_to_pkl(df, pure_pkl_file_path)
                df = posts_heavy_preprocess(df, df_list[0])
            else:
                df = users_light_preprocess(df)
                save_to_pkl(df, pure_pkl_file_path)
                df = users_heavy_preprocess(df)

            # save_to_pkl(df, pkl_file_path)
            df_list.append(df)

    return df_list[0], df_list[1]
