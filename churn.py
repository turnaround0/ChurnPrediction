#!/usr/bin/python3
import time
import argparse
import warnings
from pandas.core.common import SettingWithCopyWarning

from dataset.dataset import load_data, store_features
from features import pre, apply
from analysis import analysis
from train import train

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


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

    features_of_task1 = apply.apply_pre_features_of_task1(users_of_task1, posts_df)

    users_of_task2, posts_of_task2 = apply.apply_task2(users_df, posts_df)

    features_of_task2 = apply.apply_pre_features_of_task2(users_of_task2, posts_df)

    apply.apply_temporal_features_for_task1(features_of_task1, users_of_task1, posts_of_task1)
    apply.apply_temporal_features_for_task2(features_of_task2, users_of_task2, posts_of_task2)
    # analysis.plot_figure2(features_of_task1)

    apply.apply_frequency_features_of_task1(features_of_task1, users_of_task1, posts_of_task1)
    apply.apply_frequency_features_of_task2(features_of_task2, users_of_task2, posts_of_task2)

    # analysis.plot_figure3(features_of_task2)

    apply.apply_knowledge_features_of_task1(features_of_task1, users_of_task1, posts_of_task1, posts_df)
    apply.apply_knowledge_features_of_task2(features_of_task2, users_of_task2, posts_of_task2)

    # analysis.plot_figure4(features_of_task1)
    # store_features(features_of_task1, features_of_task2)

    apply.apply_fill_nan(features_of_task1, features_of_task2)

    train.init()
    train.table2(features_of_task1)
    train.table3(features_of_task2)
    train.figure5(features_of_task1)
    train.temporal_feature_analysis(features_of_task1)


if __name__ == '__main__':
    main()
