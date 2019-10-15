import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import tasks, temporal


def plot_figure2(features_of_task1):
    # Figure 2: Gap between posts
    #    For a user who churns, gap between consecutive posts keeps increasing.
    #    Gaps for those who stay are much lower, and stabilize around 20,000 minutes,
    #    indicating routine posting activity in every ≈2 weeks.
    list_of_K = range(1, 21)
    clist = []
    slist = []
    for K in list_of_K:
        subgroup = features_of_task1[K]
        churners_gap = []
        stayers_gap = []
        for i in range(2, K + 1):
            gapK = 'gap{}'.format(i)
            sum_gapK = list(subgroup.groupby('is_churn')[gapK].sum())
            count_gapK = list(subgroup.groupby('is_churn')[gapK].count())
            if len(sum_gapK) < 2:
                break
            churners_gap.append(sum_gapK[1] / count_gapK[1])
            stayers_gap.append(sum_gapK[0] / count_gapK[0])

        clist.append(churners_gap)
        slist.append(stayers_gap)

        # print("K={}".format(K))
        plt.plot(churners_gap, '-o', label='churner')
        plt.plot(stayers_gap, '-o', label='stayer')
        plt.legend()
        plt.axis((0, 20, 0, 15e4))
        plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    axlist = [ax1, ax2, ax3, ax4]
    for c, s, ax in zip(clist[1:], slist[1:], axlist):
        ax.plot(c, '-o', label='churner')
        ax.plot(s, '-o', label='stayer')
        ax.legend()
        ax.axis((0, 20, 0, 15e4))
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    axlist = [ax1, ax2, ax3, ax4]
    for c, s, ax in zip(clist[-4:], slist[-4:], axlist):
        ax.plot(c, '-o', label='churner')
        ax.plot(s, '-o', label='stayer')
        ax.legend()
        ax.axis((0, 20, 0, 15e4))
    plt.show()


def plot_figure2_backup(users, posts, posts_group):
    # Figure 2: Gap between posts
    #    For a user who churns, gap between consecutive posts keeps increasing.
    #    Gaps for those who stay are much lower, and stabilize around 20,000 minutes,
    #      indicating routine posting activity in every 2 weeks.
    # Draw plot for each fold about best and worst models
    # fig, axs = plt.subplots(2, 2, figsize=(6.4 * 2, 4.8 * 2))

    churn_gaps = []
    stay_gaps = []
    for K in range(2, 21):
        # ax = axs[idx]
        churn_df = tasks.getTask1Labels(users, posts_group, K)
        mean_gap_df = temporal.getTimeMeanGap(posts)
        df = churn_df.merge(mean_gap_df, how='left', left_index=True, right_index=True)

        churn_gap = df[df.is_churn == 1].gap.mean()
        stay_gap = df[df.is_churn == 0].gap.mean()

        print('GAP (churn, stay) =', churn_gap, stay_gap)

        churn_gaps.append(churn_gap)
        stay_gaps.append(stay_gap)

        fig, ax = plt.subplots()
        ax.set_title('Gap between posts ' + str(K))
        ax.set_xlabel('P where y(P) indicates gap between post P-1 and P')
        ax.set_ylabel('Mean timegap between posts (minutes)')

        x_axis = range(2, K + 1)
        ax.set_xlim(0, 21)
        ax.plot(x_axis, churn_gaps, label='Churn', marker='o')
        ax.plot(x_axis, stay_gaps, label='Stay', marker='o')
        ax.legend()
        plt.show()

    # plt.show()


def plot_figure3(features_of_task2):
    # Figure 3: # Answers vs Churn probability
    #    The probability of churning for a user decreases the more answers s/he provides.
    #    It is even lower if s/he asks more questions alongside.
    min_num_users = 50
    list_of_T = [7, 15, 30]

    for T in list_of_T:
        task2 = features_of_task2[T]
        for num_que_ask in range(5):
            subgroup = task2[task2['num_questions'] == num_que_ask]
            churn_probs = []
            num_answers = list(set(subgroup['num_answers']))
            num_answers.sort()
            for num_ans in num_answers:
                sub_subgroup = subgroup[subgroup['num_answers'] == num_ans]
                prob = sum(sub_subgroup['is_churn']) / sub_subgroup.shape[0]
                if sub_subgroup.shape[0] >= min_num_users:
                    churn_probs.append((num_ans, prob))

            plt.plot([np.log10(x[0] + 1) for x in churn_probs],
                     [np.log10(x[1] + 0.01) for x in churn_probs],
                     '-o',
                     label='{} ques asked'.format(num_que_ask))
        print("# Answers vs Churn probability")
        plt.legend()
        plt.axis((0, 2, -2, 0))
        plt.show()


def plot_figure4():
    # Figure 4: K vs Time taken for the first answer to arrive
    #    The more the time taken for a user to receive an answer,
    #      the lesser the satisfaction level and the more the chances of churning.
    pass

def plot_table2():
    # Table 2: Performance on Task 1
    seed = 1234

    for i, features in enumerate(task1_features):
        pass

def plot_table3():
    # Table 3: Performance on Task 2
    for i, features in enumerate(task2_features):
        pass


def plot_table4():
    # Table 4: Temporal Features Analysis
    for i, features in enumerate(task1_features):
        pass


def plot_table5():
    # Figure 5: Churn prediction accuracy when features from each category are used in isolation
    pass
