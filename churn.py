#!/usr/bin/python3
import argparse
import warnings
from pandas.core.common import SettingWithCopyWarning

from dataset.dataset import load_dataset, preprocess, store_features, restore_features
from features import apply
from analysis import analysis_features, analysis_train
from train import train

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description='***** Churn prediction *****')
    parser.add_argument('-d', action='store_true', help='show plots')
    parser.add_argument('-r', action='store_true', help='restore features instead of making them')
    parser.add_argument('-s', action='store', help='dataset name (tiny, small, full)')
    args = parser.parse_args()

    list_of_K = range(1, 21)
    list_of_T = [7, 15, 30]

    dataset_name = args.s if args.s else 'full'
    users_df, posts_df = load_dataset(dataset_name)
    preprocess(users_df, posts_df)

    # Featuring from dataset
    if args.r:
        features_of_task1, features_of_task2 = restore_features(list_of_K, list_of_T)
    else:
        users_of_task1, posts_of_task1 = apply.get_users_posts_of_task1(list_of_K, users_df, posts_df)
        features_of_task1 = apply.prepare_features_of_task1(list_of_K, users_of_task1, posts_df)

        users_of_task2, posts_of_task2 = apply.get_users_posts_of_task2(list_of_T, users_df, posts_df)
        features_of_task2 = apply.prepare_features_of_task2(list_of_T, users_of_task2, posts_df)

        apply.temporal_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.temporal_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        analysis_features.plot_figure2(list_of_K, features_of_task1, args.d)

        apply.frequency_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.frequency_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        analysis_features.plot_figure3(list_of_T, features_of_task2, args.d)

        apply.knowledge_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1, posts_df)
        apply.knowledge_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        analysis_features.plot_figure4(list_of_K, features_of_task1, args.d)

        apply.quality_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.quality_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        apply.consistency_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.consistency_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        apply.speed_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.speed_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        apply.gratitude_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.gratitude_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        apply.competitiveness_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.competitiveness_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        apply.content_features_of_task1(list_of_K, features_of_task1, users_of_task1, posts_of_task1)
        apply.content_features_of_task2(list_of_T, features_of_task2, users_of_task2, posts_of_task2)

        store_features(list_of_K, list_of_T, features_of_task1, features_of_task2)

    apply.fill_nan(list_of_K, list_of_T, features_of_task1, features_of_task2)

    # Training and measure performance on each task
    train.init()

    acc_models = train.performance_on_task1(list_of_K, features_of_task1)
    analysis_train.plot_table2(list_of_K, acc_models)

    acc_models = train.performance_on_task2(list_of_T, features_of_task2)
    analysis_train.plot_table3(list_of_T, acc_models)

    task1_accuracy_of_category = train.measure_task1_accuracy_of_category(list_of_K, features_of_task1)
    analysis_train.plot_figure5_of_task1(list_of_K, task1_accuracy_of_category, args.d)
    analysis_train.plot_multi_figure5_of_task1(list_of_K, task1_accuracy_of_category, args.d)

    task2_accuracy_of_category = train.measure_task2_accuracy_of_category(list_of_T, features_of_task2)
    analysis_train.plot_figure5_of_task2(list_of_T, task2_accuracy_of_category, args.d)
    analysis_train.plot_multi_figure5_of_task2(list_of_T, task2_accuracy_of_category, args.d)

    # Training and measure performance on each feature
    task1_accuracy_with_time_gap = train.performance_on_temporal(list_of_K, features_of_task1)
    analysis_train.plot_table4(task1_accuracy_with_time_gap)


if __name__ == '__main__':
    main()
