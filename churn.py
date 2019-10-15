#!/usr/bin/python3
import time
import argparse
import warnings
from pandas.core.common import SettingWithCopyWarning

from dataset.dataset import load_data
from features import pre, apply, tasks, temporal, freq
from analysis.analysis import plot_figure2

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def test_tasks(users_df, posts_df):
    a = tasks.getTask1Posts(posts_df, K=5)
    print(a)

    b = tasks.getTask1Users(users_df, posts_df, K=5)
    print(b)

    c = tasks.getTask2Posts(users_df, posts_df, T=30)
    print(c)

    d = tasks.getTask1Labels(users_df, posts_df, K=5)
    print(d)

    e = tasks.getTask2Labels(users_df, posts_df, T=30)
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

    e = temporal.getTimeMeanGap(posts_df)
    print(e)


def test_freq(users_df, posts_df, posts_group):
    a = freq.getNumAnswers(posts_group)
    print(a)

    b = freq.getNumQuestions(posts_group)
    print(b)

    c = freq.getAnsQuesRatio(a, b)
    print(c)

    d = freq.getNumPosts(posts_group)
    print(d)


def main():
    parser = argparse.ArgumentParser(description='***** Churn prediction *****')
    parser.add_argument('-d', action='store_true', help='Draw: Only loading json files and display plots')
    parser.add_argument('-s', action='store', help='Set: Config set name (ex: -s test_all')
    # args = parser.parse_args()

    # Load from dataset
    start_time = time.time()

    # You should extract the dataset for the period of the dataset: July 31, 2008 ~  July 31, 2012
    users_df, posts_df = load_data('small')

    end_time = time.time()
    print('Loading dataset time:', end_time - start_time)

    pre.apply_to_users(users_df, posts_df)
    pre.apply_to_posts(users_df, posts_df)

    users_of_task1, posts_of_task1 = apply.apply_task1(users_df, posts_df)

    print(users_of_task1)
    print(posts_of_task1)

    features_of_task1 = apply.apply_pre_features_of_task1(users_of_task1, posts_df)

    users_of_task2, posts_of_task2 = apply.apply_task2(users_df, posts_df)

    print(users_of_task2)
    print(posts_of_task2)

    features_of_task2 = apply.apply_pre_features_of_task2(users_of_task2, posts_df)
    print(features_of_task2)

    print(posts_of_task1[5])
    print(posts_of_task2[15])

    # a = know.getRepOfAcceptedAnswerer(users_df, posts_df)
    # print(a)

    # plot_figure2(users_df, posts_df, posts_group)


if __name__ == '__main__':
    main()
