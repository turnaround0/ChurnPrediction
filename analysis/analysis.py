import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import tasks, temporal

def plot_figure2(users, posts, posts_group):
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


def plot_figure3():
    # Figure 3: # Answers vs Churn probability
    #    The probability of churning for a user decreases the more answers s/he provides.
    #    It is even lower if s/he asks more questions alongside.
    for features in task2_features:
        pass


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
