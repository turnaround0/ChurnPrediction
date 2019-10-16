#!/usr/bin/python3
import argparse
import warnings
from pandas.core.common import SettingWithCopyWarning

from dataset.dataset import load_dataset, store_features, preprocess
from features import apply
from analysis import analysis
from train import train

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description='***** Churn prediction *****')
    parser.add_argument('-d', action='store_true', help='show plots')
    parser.add_argument('-s', action='store', help='Set: Config set name (ex: -s test')
    args = parser.parse_args()

    list_of_K = range(1, 21)
    list_of_T = [7, 15, 30]

    users_df, posts_df = load_dataset('full')
    preprocess(users_df, posts_df)

    users_of_task1, posts_of_task1 = apply.get_users_posts_of_task1(list_of_K, users_df, posts_df)
    features_of_task1 = apply.prepare_features_of_task1(list_of_K, users_of_task1, posts_df)

    users_of_task2, posts_of_task2 = apply.get_users_posts_of_task2(list_of_T, users_df, posts_df)
    features_of_task2 = apply.prepare_features_of_task2(list_of_T, users_of_task2, posts_df)

    apply.temporal_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
    apply.temporal_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

    analysis.plot_figure2(list_of_K, features_of_task1, args.d)
    assert()

    apply.apply_frequency_features_of_task1(features_of_task1, users_of_task1, posts_of_task1)
    apply.apply_frequency_features_of_task2(features_of_task2, users_of_task2, posts_of_task2)

    analysis.plot_figure3(features_of_task2)

    apply.apply_knowledge_features_of_task1(features_of_task1, users_of_task1, posts_of_task1, posts_df)
    apply.apply_knowledge_features_of_task2(features_of_task2, users_of_task2, posts_of_task2)

    analysis.plot_figure4(features_of_task1)
    store_features(list_of_K, list_of_T, features_of_task1, features_of_task2)

    apply.apply_fill_nan(features_of_task1, features_of_task2)

    train.init()
    train.table2(features_of_task1)
    train.table3(features_of_task2)
    train.figure5(features_of_task1)
    train.temporal_feature_analysis(features_of_task1)


if __name__ == '__main__':
    main()
