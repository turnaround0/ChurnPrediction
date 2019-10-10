import pandas as pd
import xmltodict
import os.path


def user_preprocess(df):
    # Drop unused columns and change column types
    df = df[['Id', 'Reputation', 'CreationDate', 'LastAccessDate']]

    df.CreationDate = pd.to_datetime(df.CreationDate)
    df.LastAccessDate = pd.to_datetime(df.LastAccessDate)
    df.Id = df.Id.astype('int64')
    df.Reputation = df.Reputation.astype('int64')

    df = df.set_index('Id')
    df = df.drop([-1])
    return df


def post_preprocess(df, user_df):
    df = df[['Id', 'CreationDate', 'AcceptedAnswerId', 'Score', 'OwnerUserId', 'AnswerCount', 'CommentCount']]

    df.CreationDate = pd.to_datetime(df.CreationDate)
    df.Id = df.Id.astype('int64')
    df = df.dropna(subset=['OwnerUserId'])
    df.OwnerUserId = df.OwnerUserId.astype('int64')
    df.AnswerCount = df.AnswerCount.fillna(0).astype('int64')
    df.CommentCount = df.CommentCount.fillna(0).astype('int64')

    df = df.set_index('Id')
    user_s = user_df['CreationDate']
    df = df.merge(user_s.rename('CreationDateOfOwner'), how='left', left_on='OwnerUserId', right_on='Id')

    return df


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
        if os.path.exists(pkl_file_path):
            df_list.append(load_from_pkl(pkl_file_path))
        else:
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            xml_file_path = data_path + dataset_name + '.xml'

            df = xml2df(xml_file_path)

            if dataset_name == 'Posts':
                df = post_preprocess(df, df_list[0])
            else:
                df = user_preprocess(df)

            # Cut dataset by given period
            df = df[(df['CreationDate'] >= start_time) & (df['CreationDate'] <= end_time)]

            # save_to_pkl(df, pkl_file_path)
            df_list.append(df)

    return df_list[0], df_list[1]
