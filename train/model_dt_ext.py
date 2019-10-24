import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeExtClassifier:
    def __init__(self, *args, **kwargs):
        self.max_round = kwargs.pop('max_round')
        self.p_value = kwargs.pop('p_value')
        self.kwargs = kwargs
        self.args = args

    def fit(self, src_x, src_y):
        fit_list = []
        u_list = []
        rest = len(src_y)

        x = src_x.copy().reset_index(drop=True)
        y = src_y.copy().reset_index(drop=True)

        full_x, full_y, = x, y
        for idx in range(self.max_round):
            if idx == self.max_round - 1:
                if idx == 0:
                    fit_list.append(self.model.fit(full_x, full_y))
                else:
                    fit_list.append(fit_list[0])
                ext_len = rest
                u_value = 1
            else:
                model = DecisionTreeClassifier(*self.args, **self.kwargs).fit(x, y)
                fit_list.append(model)

                pred = model.predict_proba(x)
                df_pred = pd.DataFrame(pred)
                df_pred.columns = ['false', 'true']

                criterion = (df_pred.true * df_pred.false).to_frame()
                criterion.columns = ['criterion']
                df_ext = criterion.sort_values('criterion', ascending=True)

                cut_len = round(rest * self.p_value)
                df_cut_ext = df_ext.iloc[: cut_len]
                u_value = df_cut_ext.iloc[-1].criterion

                df_rest_ext = df_ext.iloc[cut_len:]
                df_rest_ext = df_rest_ext[df_rest_ext.criterion == u_value]
                df_cut_ext = pd.concat([df_cut_ext, df_rest_ext])

                x = x.drop(df_cut_ext.index).reset_index(drop=True)
                y = y.drop(df_cut_ext.index).reset_index(drop=True)
                ext_len = len(df_cut_ext)

            rest -= ext_len
            u_list.append(u_value)
            # print('Train) Round, u value, rest, number of extraction:', idx, u_value, rest, ext_len)

        return DecisionTreeExtModel(self.max_round, self.p_value, zip(fit_list, u_list))


class DecisionTreeExtModel:
    def __init__(self, max_round, p_value, obj):
        self.max_round = max_round
        self.p_value = p_value
        self.obj = obj

    def predict(self, x):
        meet_list = []
        final_pred = None
        rest = len(x)
        for idx, (model, u_value) in enumerate(self.obj):
            pred = model.predict_proba(x)
            df_pred = pd.DataFrame(pred)
            df_pred.columns = ['false', 'true']
            meet = (df_pred.true * df_pred.false <= u_value)

            if idx == 0:
                final_pred = df_pred.true
                df_pred.true[~meet] = None
            else:
                for prev_idx in range(idx):
                    prev_meet = meet_list[prev_idx]
                    meet[prev_meet] = False
                final_pred[meet] = df_pred.true[meet]

            meet_list.append(meet)
            if idx == self.max_round - 1:
                rest = 0
            else:
                rest -= meet.sum()
            # print('Prediction) Round, u value, rest, meet count:', idx, u_value, rest, meet.sum())

        final_pred[final_pred >= 0.5] = 1
        final_pred[final_pred < 0.5] = 0

        return final_pred.to_numpy()
