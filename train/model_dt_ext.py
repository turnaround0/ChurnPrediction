import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeExtClassifier:
    def __init__(self, **kwargs):
        self.max_round = kwargs.pop('max_round')
        self.p_value = kwargs.pop('p_value')
        self.kwargs = kwargs
        self.model = DecisionTreeClassifier()

    def fit(self, x, y):
        fit_list = []
        u_list = [1]

        full_x, full_y, = x, y
        fit_list.append(self.model.fit(full_x, full_y, **self.kwargs))

        return DecisionTreeExtModel(self.max_round, self.p_value, self.kwargs, zip(fit_list, u_list))

    def fit2(self, x, y):
        fit_list = []
        u_list = []
        rest = len(y)

        full_x, full_y, = x, y
        for idx in range(ext_params['max_round']):
            if idx == ext_params['max_round'] - 1:
                if idx == 0:
                    fit_list.append(self.model.fit(full_x, full_y, **self.kwargs))
                else:
                    fit_list.append(fit_list[0])
                ext_idx_list = x.index.tolist()
                u_value = 0
                rest = 0
            else:
                fit_list.append(self.model.fit(x, y, **self.kwargs))
                df_ext = pd.DataFrame(ext_list).sort_values('abs_uplift', ascending=False)

                if len(df_ext) == 1:
                    # If there is only one group after building tree, it should be halted.
                    u_value = 0
                    rest = 0
                    if idx > 0:
                        fit_list.pop()
                        fit_list.append(fit_list[0])
                    ext_idx_list = x.index.tolist()
                    u_list.append(u_value)
                    print('Before max round, tree has only one group.')
                    print('Train) Round, u value, rest, number of extraction:', idx, u_value, rest, len(ext_idx_list))
                    break

                df_ext['n_cumsum_samples'] = df_ext['n_samples'].cumsum()
                cut_len = rest * p_value
                cut_len_upper = df_ext[df_ext['n_cumsum_samples'] > cut_len]['n_cumsum_samples'].iloc[0]
                df_cut_ext = df_ext[df_ext['n_cumsum_samples'] <= cut_len_upper]
                if len(df_cut_ext) == len(df_ext):
                    # Should not extract all data from training set
                    df_cut_ext = df_ext.iloc[: -1]
                u_value = df_cut_ext.iloc[-1]['abs_uplift']

                ext_idx_list = df_cut_ext['idx_list'].sum()
                x = x.drop(ext_idx_list)
                y = y.drop(ext_idx_list)
                rest -= len(ext_idx_list)

            u_list.append(u_value)
            print('Train) Round, u value, rest, number of extraction:', idx, u_value, rest, len(ext_idx_list))

        return zip(fit_list, u_list)


class DecisionTreeExtModel:
    def __init__(self, max_round, p_value, kwargs, obj):
        self.max_round = max_round
        self.p_value = p_value
        self.kwargs = kwargs
        self.obj = obj

    def predict(self, x):
        pred = None
        for idx, (model_fit, u_value) in enumerate(self.obj):
            pred = model_fit.predict(x)

        return pred

    def predict2(self, x, **kwargs):
        kwargs.update({'method': 'ed'})

        meet_list = []
        final_pred = None
        rest = len(x)
        for idx, (model_fit, u_value) in enumerate(self.obj):
            pred = model_dt.predict(model_fit, x, **kwargs)
            meet = pd.Series(np.abs(pred['pr_y1_t1'] - pred['pr_y1_t0']) >= u_value)

            if idx == 0:
                final_pred = pred
                final_pred[~meet] = None
            else:
                for prev_idx in range(idx):
                    prev_meet = meet_list[prev_idx]
                    meet[prev_meet] = False
                final_pred[meet] = pred[meet]

            meet_list.append(meet)
            if idx == ext_params['max_round'] - 1:
                rest = 0
            else:
                rest -= meet.sum()
            print('Prediction) Round, u value, rest, meet count:', idx, u_value, rest, meet.sum())

        return final_pred
