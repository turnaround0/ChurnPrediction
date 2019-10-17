import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_table2(list_of_K, acc_models):
    columns = ['k(posts)'] + list(acc_models.keys())
    lines = []
    for K in list_of_K:
        lines.append([K] + [acc_models[model_name][K] for model_name in acc_models.keys()])
    df = pd.DataFrame(lines, columns=columns).set_index('k(posts)')
    print(df)
    df.to_csv('output/table2.csv')


def plot_table3(list_of_T, acc_models):
    columns = ['T(days)'] + list(acc_models.keys())
    lines = []
    for T in list_of_T:
        lines.append([T] + [acc_models[model_name][T] for model_name in acc_models.keys()])
    df = pd.DataFrame(lines, columns=columns).set_index('T(days)')
    print(df)
    df.to_csv('output/table3.csv')


def plot_table4(task1_accuracy_with_time_gap):
    # Table of temporal gap features analysis
    columns = ['k', 'Only gapK (Temporal Gaps)', 'Only last_gap (Last-Gap)']
    lines = []
    for K, acc in task1_accuracy_with_time_gap.items():
        lines.append([K] + acc)
    df = pd.DataFrame(lines, columns=columns).set_index('k')
    print(df)
    df.to_csv('output/table4.csv')


def plot_figure5_of_task1(list_of_K, task1_accuracy_of_category, is_display=False):
    for title, predictions in task1_accuracy_of_category.items():
        if len(predictions) == 0:
            continue
        n_groups = len(list_of_K)
        index = np.arange(n_groups)

        fig, ax = plt.subplots()
        ax.bar(index, predictions, tick_label=list_of_K, align='center')
        ax.set_title(title)
        ax.set_xlim(-1, n_groups)
        ax.set_ylim(50, 80)
        fig.savefig('output/figure5_task1_{}.png'.format(title))
        if is_display:
            plt.show()
        plt.close(fig)


def plot_figure5_of_task2(list_of_T, task2_accuracy_of_category, is_display=False):
    for title, predictions in task2_accuracy_of_category.items():
        if len(predictions) == 0:
            continue
        n_groups = len(list_of_T)
        index = np.arange(n_groups)

        fig, ax = plt.subplots()
        ax.bar(index, predictions, tick_label=list_of_T, align='center')
        ax.set_title(title)
        ax.set_xlim(-1, n_groups)
        ax.set_ylim(50, 80)
        fig.savefig('output/figure5_task2_{}.png'.format(title))
        if is_display:
            plt.show()
        plt.close(fig)


def plot_multi_figure5_of_task1(list_of_K, task1_accuracy_of_category, is_display=False):
    fig, axs = plt.subplots(2, 5, figsize=(6.4 * 4, 4.8 * 2))
    ax_list = [ax for sub_axs in axs for ax in sub_axs]

    for idx, (title, predictions) in enumerate(task1_accuracy_of_category.items()):
        if len(predictions) == 0:
            continue
        ax = ax_list[idx]
        n_groups = len(list_of_K)
        index = np.arange(n_groups)

        ax.bar(index, predictions, tick_label=list_of_K, align='center')
        ax.set_title(title)
        ax.set_xlim(-1, n_groups)
        ax.set_ylim(50, 80)

    fig.suptitle('Classification accuracy for each feature class')
    fig.text(0.5, 0.04, 'Number of observation posts (K)', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    fig.savefig('output/figure5_multi_task1.png')
    if is_display:
        plt.show()
    plt.close(fig)


def plot_multi_figure5_of_task2(list_of_T, task2_accuracy_of_category, is_display=False):
    fig, axs = plt.subplots(2, 5, figsize=(6.4 * 4, 4.8 * 2))
    ax_list = [ax for sub_axs in axs for ax in sub_axs]

    for idx, (title, predictions) in enumerate(task2_accuracy_of_category.items()):
        if len(predictions) == 0:
            continue
        ax = ax_list[idx]
        n_groups = len(list_of_T)
        index = np.arange(n_groups)

        ax.bar(index, predictions, tick_label=list_of_T, align='center')
        ax.set_title(title)
        ax.set_xlim(-1, n_groups)
        ax.set_ylim(50, 80)

    fig.suptitle('Classification accuracy for each feature class')
    fig.text(0.5, 0.04, 'Observation period in days (T)', ha='center')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    fig.savefig('output/figure5_multi_task2.png')
    if is_display:
        plt.show()
    plt.close(fig)
