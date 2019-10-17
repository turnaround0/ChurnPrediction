import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def decision_tree_classifier_with_args(*arg, **kwargs):
    kwargs['random_state'] = 1234
    return DecisionTreeClassifier(*arg, **kwargs)


def logistic_regression_with_args(*arg, **kwargs):
    kwargs['max_iter'] = 1e3
    kwargs['solver'] = 'lbfgs'
    kwargs['random_state'] = 1234
    return LogisticRegression(*arg, **kwargs)


def svc_with_args(*args, **kwargs):
    kwargs['max_iter'] = 1e3
    kwargs['gamma'] = 'auto'
    kwargs['kernel'] = 'rbf'
    kwargs['random_state'] = 1234
    return SVC(*args, **kwargs)


def linear_svc_with_args(*args, **kwargs):
    kwargs['max_iter'] = 1e3
    kwargs['random_state'] = 1234
    return LinearSVC(*args, **kwargs)
