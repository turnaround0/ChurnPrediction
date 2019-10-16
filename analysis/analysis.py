import numpy as np
import matplotlib.pyplot as plt

from features import tasks, temporal


def plot_single_figure2(list_of_K, features_of_task1, is_display):
    churn_list, stay_list = [], []

    for K in list_of_K:
        subgroup = features_of_task1[K]
        churners_gap, stayers_gap = [], []

        for i in range(2, K + 1):
            gapK = 'gap{}'.format(i)
            mean_gapK = list(subgroup.groupby('is_churn')[gapK].mean())
            if len(mean_gapK) < 2:
                break
            churners_gap.append(mean_gapK[1])
            stayers_gap.append(mean_gapK[0])

        churn_list.append(churners_gap)
        stay_list.append(stayers_gap)

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        x_axis = range(2, K + 1)
        ax.set_title('Gap between posts (K = ' + str(K) + ')')
        ax.set_xlabel('P where y(P) indicates gap between post P-1 and P')
        ax.set_ylabel('Mean time gap between posts (minutes)')
        ax.plot(x_axis, churners_gap, '-o', label='churner')
        ax.plot(x_axis, stayers_gap, '-o', label='stayer')
        ax.xaxis.set_ticks(range(0, 21, 5))
        ax.grid(linestyle=':')
        ax.legend()
        ax.axis((0, 21, 0, 15e4))
        fig.savefig('output/figure2_gap{}.png'.format(K))
        if is_display:
            plt.show()
        plt.close(fig)

    return churn_list, stay_list


def plot_multi_figure2(plot_type, churn_list, stay_list, is_display):
    fig, axs = plt.subplots(2, 2, figsize=(6.4 * 2, 4.8 * 2))
    ax_list = [ax for sub_axs in axs for ax in sub_axs]

    if plot_type == 'first':
        plot_zip = zip(churn_list[1:], stay_list[1:], ax_list)
    else:   # If plot_type is 'last'
        plot_zip = zip(churn_list[-4:], stay_list[-4:], ax_list)

    for churners, stayers, ax in plot_zip:
        x_axis = range(2, len(churners) + 2)
        ax.plot(x_axis, churners, '-o', label='churner')
        ax.plot(x_axis, stayers, '-o', label='stayer')
        ax.xaxis.set_ticks(range(0, 21, 5))
        ax.grid(linestyle=':')
        ax.legend()
        ax.axis((0, 21, 0, 15e4))

    fig.suptitle('Gap between posts')
    fig.text(0.5, 0.04, 'P where y(P) indicates gap between post P-1 and P', ha='center')
    fig.text(0.04, 0.5, 'Mean time gap between posts (minutes)', va='center', rotation='vertical')
    fig.savefig('output/figure2_gap_{}.png'.format(plot_type))
    if is_display:
        plt.show()
    plt.close(fig)


def plot_figure2(list_of_K, features_of_task1, is_display=False):
    # Figure 2: Gap between posts
    #    For a user who churns, gap between consecutive posts keeps increasing.
    #    Gaps for those who stay are much lower, and stabilize around 20,000 minutes,
    #    indicating routine posting activity in every ≈2 weeks.
    churn_list, stay_list = plot_single_figure2(list_of_K, features_of_task1, is_display)

    plot_multi_figure2('first', churn_list, stay_list, is_display)
    plot_multi_figure2('last', churn_list, stay_list, is_display)


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


def plot_figure4(features_of_task1):
    # Figure 4: K vs Time taken for the first answer to arrive
    #  The more the time taken for a user to receive an answer,
    #  the lesser the satisfaction level and the more the chances of churning.
    list_of_K = range(1, 21)
    churners_time = []
    stayers_time = []
    for K in list_of_K:
        subgroup = features_of_task1[K]
        churners = subgroup[subgroup['is_churn'] == 1][subgroup['time_for_first_ans'] > 0]
        stayers = subgroup[subgroup['is_churn'] == 0][subgroup['time_for_first_ans'] > 0]
        churners_time.append(churners['time_for_first_ans'].mean())
        stayers_time.append(stayers['time_for_first_ans'].mean())

    plt.plot(churners_time, '-o', label='churner')
    plt.plot(stayers_time, '-o', label='stayer')
    plt.legend()
    # plt.axis((0,20,8e3,22e3))
    plt.show()


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
