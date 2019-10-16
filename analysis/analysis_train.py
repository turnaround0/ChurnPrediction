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


def figure5_of_task1(list_of_K, task1_accuracy_of_category, is_display=False):
    for title, predictions in task1_accuracy_of_category.items():
        if len(predictions) == 0:
            continue
        n_groups = len(list_of_K)
        index = np.arange(n_groups)

        fig, ax = plt.subplots()
        ax.bar(index, predictions, tick_label=list_of_K, align='center')
        ax.set_title(title)
        ax.set_xlim(-1, n_groups)
        ax.set_ylim(40, 100)
        fig.savefig('output/figure5_task1_{}.png'.format(title))
        if is_display:
            plt.show()
        plt.close(fig)


def figure5_of_task2(list_of_T, task2_accuracy_of_category, is_display=False):
    for title, predictions in task2_accuracy_of_category.items():
        if len(predictions) == 0:
            continue
        n_groups = len(list_of_T)
        index = np.arange(n_groups)

        fig, ax = plt.subplots()
        ax.bar(index, predictions, tick_label=list_of_T, align='center')
        ax.set_title(title)
        ax.xlim(-1, n_groups)
        ax.ylim(40, 100)
        fig.savefig('output/figure5_task2_{}.png'.format(title))
        if is_display:
            plt.show()
        plt.close(fig)
