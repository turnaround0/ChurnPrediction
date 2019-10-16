from sklearn.linear_model import LogisticRegression


def LogisticRegressionWithArgs(*arg, **kwarg):
    kwarg['max_iter'] = 1e3
    # kwarg['solver'] = 'lbfgs'
    kwarg['tol'] = 1e2
    return LogisticRegression(*arg, **kwarg)
