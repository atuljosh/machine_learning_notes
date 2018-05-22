import numpy as np
import patsy as ptsy
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from glm import LinearModel

# TODO Write valid unittests

def generate_data(n, loss, wt_param=2, return_rate=False):
    """
    Generate random data for testing
    :param n:
    :param loss:
    :param wt_param:
    :return:
    """
    w = np.random.randint(1, wt_param, n)
    if loss == 'squared':
        y = np.random.normal(50, 100, size=n)
    if loss == 'logistic':
        # Binomial - n numper of trials, p probability [p_or_label trials] - response
        if return_rate:
            y = [sum(np.random.binomial(1, 0.1, 100) == 1) / 100.0 for _ in xrange(n)]
        else:
            y = np.random.binomial(1, 0.1, n)
    if loss == 'poisson':
        # Poisson - lambda expected events in an interval [avg_no_of_events_or_rate trials] - response
        if return_rate:
            y = [count*1.0/trials for count, trials in zip(np.random.poisson(10, size=n), w)]
        else:
            y = np.random.poisson(10, size=n)

    d = {'value': y,
         'feature1': [np.random.choice(['a', 'b', 'c']) for _ in xrange(n)],
         'feature2': [np.random.choice(['pp', 'qq']) for _ in xrange(n)]}

    df = pd.DataFrame(d)
    out = ptsy.dmatrices('value ~ feature1 + feature2', data=df, return_type='dataframe')
    y, X = out
    return w, y, X

def test_linear_model_without_regularization(n=100):
    print '----- Linear model without weights and regularization ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='squared')
    # Our simple
    glm_simple = LinearModel(loss='squared')
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res

    # Statmodels
    glm = sm.GLM(y, X, family=sm.families.Gaussian())
    res = glm.fit(method='qr')
    test_results['sm_glm'] = res.params.values.tolist()
    summary = res.summary()
    # Sklearn
    sk_glm = linear_model.LinearRegression(fit_intercept=False) # Since design matrix has added intercept
    sk_glm.fit(X, y)
    test_results['sklearn_glm'] = sk_glm.coef_.ravel().tolist()
    print test_results

def test_logistic_model_without_regularization_no_rate(n=100):
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='logistic', wt_param=1000, return_rate=False)
    # Statmodels
    y = np.array(y).ravel()
    # weights = np.array(weights).ravel().astype('int')
    # y = np.multiply(y, weights).astype('int')
    # non_actions = np.subtract(weights, y).astype('int')
    # y_sm = np.array(zip(y, non_actions))
    glm = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm.fit()
    test_results['sm_glm'] = res.params
    # Sklearn (Expects labels not probabilities)
    sk_glm = linear_model.LogisticRegression(fit_intercept=False) # Since design matrix has added intercept
    sk_glm.fit(X, y)
    test_results['sklearn_glm'] = sk_glm.coef_.ravel().tolist()
    print test_results

def test_weighted_logistic_model_without_regularization_with_rate(n=100):
    print '----- Weighted Logistic model without regularization and response as rate (response between 0 and 1) ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='logistic', wt_param=1000, return_rate=True)
    # Our simple
    glm_simple = LinearModel(loss='logistic')
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res

    # Statmodels
    y = np.array(y).ravel()
    weights = np.array(weights).ravel().astype('float')
    y = np.multiply(y, weights).astype('float')
    non_actions = np.subtract(weights, y).astype('float')
    y_sm = np.array(zip(y, non_actions))
    glm = sm.GLM(y_sm, X, family=sm.families.Binomial())
    res = glm.fit()
    test_results['sm_glm'] = res.params.ravel().tolist()

    print test_results

def test_poisson_model_without_regularization_with_count(n=100):
    print '----- Poisson model without weights and regularization ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='poisson')

    # Our simple
    glm_simple = LinearModel(loss='poisson')
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res

    # Statmodels (poisson strictly expect count data, out of 1 trial)
    y = np.array(y).ravel()
    glm = sm.GLM(y, X, family=sm.families.Poisson())
    res = glm.fit()
    test_results['sm_glm'] = res.params.ravel().tolist()
    print test_results

def test_weighted_linear_model_without_regularization(n=100):
    print '----- Weighted Linear model without regularization ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='squared', wt_param=10)

    # Our simple
    glm_simple = LinearModel(loss='squared')
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res
    # Statmodels
    wls = sm.WLS(y,X, weights=weights)
    res1 = wls.fit()
    test_results['sm_wls'] = res1.params.values.tolist()
    # Sklearn
    sk_glm = linear_model.LinearRegression(fit_intercept=False) # Since design matrix has added intercept
    sk_glm.fit(X, y, sample_weight=weights)
    test_results['sklearn_glm'] = sk_glm.coef_.ravel().tolist()
    print test_results

# def test_weighted_logistic_model_without_regularization(n=100):
#     print '----- Weighted Logistic model without regularization ------'
#     d = dict()
#     test_results = pd.DataFrame(d.items())
#     weights, y, X = generate_data(n, loss='logistic', wt_param=1000, return_rate=True)
#     # Our simple
#     glm_simple = LinearRegression(loss='logistic')
#     glm_simple_res = glm_simple.fit(X, y, weights)
#     test_results['simple_glm'] = glm_simple_res
#
#     # Statmodels
#     y = np.array(y).ravel()
#     weights = np.array(weights).ravel().astype('float')
#     y = np.multiply(y, weights).astype('float')
#     non_actions = np.subtract(weights, y).astype('float')
#     y_sm = np.array(zip(y, non_actions))
#     glm = sm.GLM(y_sm, X, family=sm.families.Binomial())
#     res = glm.fit()
#     test_results['sm_glm'] = res.params.ravel().tolist()
#     print test_results

def test_weighted_poisson_model_without_regularization_with_count(n=100):
    print '----- Poisson model with weights and without regularization ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='poisson', wt_param=10, return_rate=True)

    # Our simple
    glm_simple = LinearModel(loss='poisson')
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res

    # Nothing to test / compare to :
    # Probably Best thing to compare to is statmodels logistic with scaled response
    print test_results

def test_weighted_linear_model_with_regularization(n=100):
    print '----- Weighted Linear model WITH regularization ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='squared', wt_param=10)

    # Match prior mean and prior var
    # Pass different means here e.g.
    prior_mean = {}
    prior_var = {}
    for i in X.columns:
        prior_mean[i] = 0
        prior_var[i] = 0.000001
    prior_mean["Intercept"] = np.mean(y.values)

    # Our simple
    glm_simple = LinearModel(loss='squared', regularize_intercept=True, prior_mean=prior_mean, prior_var=prior_var)
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res
    print prior_mean
    print test_results

def test_weighted_logistic_model_with_regularization(n=100):
    print '----- Weighted Logistic model WITH regularization and response as rate (response between 0 and 1) ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='logistic', wt_param=1000, return_rate=True)

    # Match prior mean and prior var
    # Pass different means here e.g.
    prior_mean = {}
    prior_var = {}
    for i in X.columns:
        prior_mean[i] = 0
        prior_var[i] = 0.000001
    prior_mean["Intercept"] = np.mean(y.values)

    # Our simple
    glm_simple = LinearModel(loss='logistic', regularize_intercept=True, prior_mean=prior_mean, prior_var=prior_var)
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res

    print prior_mean
    print test_results

def test_weighted_poisson_model_with_regularization(n=100):
    print '----- Poisson model with weights and without regularization ------'
    d = dict()
    test_results = pd.DataFrame(d.items())
    weights, y, X = generate_data(n, loss='poisson', wt_param=10, return_rate=True)

    prior_mean = {}
    prior_var = {}
    for i in X.columns:
        prior_mean[i] = 0
        prior_var[i] = 0.000001
    prior_mean["Intercept"] = np.mean(y.values)

    # Our simple
    glm_simple = LinearModel(loss='poisson', regularize_intercept=True, prior_mean=prior_mean, prior_var=prior_var)
    glm_simple_res = glm_simple.fit(X, y, weights)
    test_results['simple_glm'] = glm_simple_res

    # Nothing to test / compare to :
    # Probably Best thing to compare to is statmodels logistic with scaled response
    print prior_mean
    print test_results


if __name__ == '__main__':
    test_linear_model_without_regularization(n=100)
    test_weighted_linear_model_without_regularization(n=100)
    #### test_logistic_model_without_regularization_no_rate(n=100) # Not done yet
    test_weighted_logistic_model_without_regularization_with_rate(n=100)
    test_poisson_model_without_regularization_with_count(n=100)
    test_weighted_poisson_model_without_regularization_with_count(n=100)
    test_weighted_linear_model_with_regularization(n=100)
    test_weighted_logistic_model_with_regularization(n=100)
    test_weighted_poisson_model_with_regularization(n=100)