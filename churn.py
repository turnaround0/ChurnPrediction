#!/usr/bin/python3
import time
import argparse
import warnings
from pandas.core.common import SettingWithCopyWarning

from dataset.dataset import load_data
from features import tasks, temporal, freq

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def test_tasks(users_df, posts_df, posts_group):
    a = tasks.getTask1Posts(posts_group, K=5)
    print(a)

    b = tasks.getTask1Users(users_df, posts_group, K=5)
    print(b)

    c = tasks.getTask2Posts(users_df, posts_df, posts_group, T=30)
    print(c)

    d = tasks.getTask1Labels(users_df, posts_group, K=5)
    print(d)

    e = tasks.getTask2Labels(users_df, posts_df, posts_group, T=30)
    print(e)


def test_temporal(users_df, posts_df, posts_group):
    a = temporal.getTimeGap1OfUser(posts_group)
    print(a)

    b = temporal.getTimeGapsOfPosts(posts_group, 3)
    print(b)

    c = temporal.getTimeLastGapOfPosts(posts_group)
    print(c)

    d = temporal.getTimeSinceLastPost(posts_group, '2012-07-31')
    print(d)


def main():
    parser = argparse.ArgumentParser(description='***** Churn prediction *****')
    parser.add_argument('-d', action='store_true', help='Draw: Only loading json files and display plots')
    parser.add_argument('-s', action='store', help='Set: Config set name (ex: -s test_all')
    # args = parser.parse_args()

    # Load from dataset
    start_time = time.time()

    # You should extract the dataset for the period of the dataset: July 31, 2008 ~  July 31, 2012
    users_df, posts_df = load_data('tiny', '2008-07-31', '2012-07-31')

    end_time = time.time()
    print('Loading dataset time:', end_time - start_time)

    # Grouping posts dataframe by user id
    start_time = time.time()

    # Get posts group by user id
    # Group should be reused for avoiding regenerating the group.
    posts_group = posts_df.groupby('OwnerUserId')

    end_time = time.time()
    print('Grouping time:', end_time - start_time)

    a = freq.getNumAnswers(posts_group)
    print(a)


if __name__ == '__main__':
    main()
