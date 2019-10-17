from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC


def logistic_regression_with_args(*arg, **kwargs):
    kwargs['max_iter'] = 1e3
    # kwargs['solver'] = 'saga'
    kwargs['tol'] = 1e2
    kwargs['random_state'] = 1234
    return LogisticRegression(*arg, **kwargs)


def svc_with_args(*args, **kwargs):
    kwargs['max_iter'] = 1e3
    kwargs['tol'] = 1e2
    kwargs['random_state'] = 1234
    return SVC(*args, **kwargs)


def linear_svc_with_args(*args, **kwargs):
    kwargs['max_iter'] = 1e3
    kwargs['tol'] = 1e2
    kwargs['random_state'] = 1234
    return LinearSVC(*args, **kwargs)
