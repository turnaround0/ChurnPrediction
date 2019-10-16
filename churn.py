#!/usr/bin/python3
import argparse
import warnings
from pandas.core.common import SettingWithCopyWarning

from dataset.dataset import load_dataset, preprocess, store_features, restore_features
from features import apply
from analysis import analysis
from train import train

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description='***** Churn prediction *****')
    parser.add_argument('-d', action='store_true', help='show plots')
    parser.add_argument('-r', action='store_true', help='restore features instead of making them')
    args = parser.parse_args()

    list_of_K = range(1, 21)
    list_of_T = [7, 15, 30]

    users_df, posts_df = load_dataset('small')
    preprocess(users_df, posts_df)

    if args.r:
        features_of_task1, features_of_task2 = restore_features(list_of_K, list_of_T)
    else:
        users_of_task1, posts_of_task1 = apply.get_users_posts_of_task1(list_of_K, users_df, posts_df)
        features_of_task1 = apply.prepare_features_of_task1(list_of_K, users_of_task1, posts_df)

        users_of_task2, posts_of_task2 = apply.get_users_posts_of_task2(list_of_T, users_df, posts_df)
        features_of_task2 = apply.prepare_features_of_task2(list_of_T, users_of_task2, posts_df)

        apply.temporal_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.temporal_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        analysis.plot_figure2(list_of_K, features_of_task1, args.d)

        apply.frequency_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.frequency_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        analysis.plot_figure3(list_of_T, features_of_task2, args.d)

        apply.knowledge_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1, posts_df)
        apply.knowledge_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        analysis.plot_figure4(list_of_K, features_of_task1, args.d)

        apply.fill_nan(list_of_K, list_of_T, features_of_task1, features_of_task2)
        store_features(list_of_K, list_of_T, features_of_task1, features_of_task2)

    train.init()
    train.performance_on_task1(list_of_K, features_of_task1)
    train.table3(list_of_T, features_of_task2)
    train.figure5(list_of_K, features_of_task1)
    train.temporal_feature_analysis(list_of_K, features_of_task1)


if __name__ == '__main__':
    main()
