import numpy as np
import math
import pandas as pd
import scipy.stats as stats


class GLM(object):
    """
    A simple class for implementing bayesian weighted
    generalized linear models (support for gaussian, logistic and poisson case)
    """

    def __init__(self,
                 loss="squared",
                 max_iter=100,
                 regularize_intercept=True,
                 regularization='l2',
                 prior_mean=None,
                 prior_var=None,
                 solver='qr'):
        self.k = None
        self.n = None
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.loss = loss
        self.max_iter = max_iter
        self.regularize_intercept = regularize_intercept
        self.regularization = regularization
        self.ordered_feature_list = None
        self.solver = solver

        # Save model
        self.beta = None
        self.t_scores = None
        self.p_values = None
        self.beta_sd = None

    def _data_sanity_checks(self):
        # check for design matrix as numpy array
        # check for correct formats of prior mean and prior variance
        pass

    def _set_variables(self, X):
        self.k = len(X.columns)
        self.n = len(X)

    def _get_x_star(self, X):
        if self.regularize_intercept:
            x_add = pd.DataFrame(np.diag(np.ones(self.k)))
        else:
            x_add = pd.DataFrame(np.diag(np.ones(self.k-1)))
        x_add.columns = X.columns
        X = X.append(x_add, ignore_index=True)
        return X

    def _set_w(self, w):
        if len(w) > 0:
            w = w.ravel()
        else:
            w = np.ones(self.n)
        return w

    def _get_y_star(self, X, y):
        if self.prior_mean:
            beta_add = pd.DataFrame([self.prior_mean[feature] if feature in self.prior_mean else 0 for feature in X.columns])
            beta_add.columns = y.columns
            y = y.append(beta_add, ignore_index=True)
        return y

    def _get_w_diag(self, var_y, w):

        var_diag = var_y / np.sqrt(w)
        # weigh output variance and append prior var vector
        if self.prior_var:
            prior_var_vec = np.array(self.prior_var.values())
            var_diag = np.concatenate((var_diag, prior_var_vec), axis=0)
        w_diag = 1.0 / var_diag

        return w_diag

    def _get_pseudo_w(self, fitted_values, weights_count):
        if self.loss == 'squared':
            return weights_count

        # weights are nothing but variance of binomial = npq
        if self.loss == 'logistic':
            fit_prob = map(self._logit_inverse, fitted_values)
            ones = np.ones(len(fit_prob))
            fit_prob_cpl = np.subtract(ones, np.array(fit_prob))
            pq = np.multiply(fit_prob, fit_prob_cpl)
            pseudo_wt = np.multiply(weights_count, pq)
            return pseudo_wt

        # variance of poisson = np
        if self.loss == 'poisson':
            fit_rate = np.array(map(math.exp, fitted_values))
            pseudo_wt = np.multiply(weights_count, fit_rate)
            return pseudo_wt

    def _get_pseudo_y(self, fitted_values, y):
        if self.loss == 'gaussian':
            return y

        # pseudo y = fit_val + (y - fit prob)/(pseudo w)
        if self.loss == 'logistic':
            fit_prob = map(self._logit_inverse, fitted_values)
            ones = np.ones(len(fit_prob))
            fit_prob = np.array(fit_prob).ravel()
            y_raw = np.array(y).ravel()
            pred_error = np.subtract(y_raw, fit_prob)

            fit_prob_cpl = np.subtract(ones, np.array(fit_prob))
            pq = np.multiply(fit_prob, fit_prob_cpl)

            add_term = np.divide(pred_error, pq)
            y_mod = np.add(fitted_values, add_term)
            y_mod = pd.DataFrame(y_mod)
            y_mod.columns = y.columns
            return y_mod

        # pseudo y = fit_val + (y - fit rate)/(fit rate)
        if self.loss == 'poisson':
            fit_rate = map(math.exp, fitted_values)
            fit_rate = np.array(fit_rate).ravel()
            y_raw = np.array(y).ravel()
            pred_error = np.subtract(y_raw, fit_rate)
            pred_error_normalized = np.divide(pred_error, fit_rate)
            y_mod = np.add(fitted_values, pred_error_normalized)
            y_mod = pd.DataFrame(y_mod)
            y_mod.columns = y.columns
            return y_mod

    def _get_starting_values(self, y):
        """
        Function to return starting values for beta
        Starting values are assumed as:
        intercept = mean and 0 for other coefficients
        gaussian - y : real values
        logistic - y : probabilities & hence logit (what if design matrix is labels, count successes?)
        poisson - y : rates & hence log values (what if design matrix is counts)
        """
        if self.loss == 'logistic':
            y = [self._logit(x) for x in y.values.ravel()]

        elif self.loss == 'poisson':
            y = [math.log(x) for x in y.values.ravel()]

        intercept_start = np.mean(y)
        coeff = np.zeros(self.k - 1)
        full_beta = np.append(intercept_start, coeff)
        full_beta = pd.Series(full_beta)
        return intercept_start, full_beta

    def _get_new_beta(self, q, r, y_star, w_star):
        if self.solver != 'qr':
            print 'Solver not supported yet'
        y_input = y_star.mul(w_star, axis=0)
        y_new_input = np.dot(q.T, y_input)
        new_beta = np.linalg.solve(r, y_new_input)
        new_beta = pd.Series(new_beta.flatten())
        return new_beta

    def _get_qr(self, x_star, w_star):
        if self.solver != 'qr':
            print 'Solver not supported yet'
        qr_input = x_star.mul(w_star, axis=0)
        q, r = np.linalg.qr(qr_input)
        return q, r

    @staticmethod
    def _logit(z):
        return math.log(z) - math.log(1 - z)

    @staticmethod
    def _logit_inverse(z):
        if z > 0:
            return 1. / (1. + np.exp(-z))
        elif z <= 0:
            return np.exp(z) / (1 + np.exp(z))

    def _if_converged(self, old_beta, new_beta, tol=0.0001):
        is_zero = old_beta.abs() < tol
        diff = (old_beta - new_beta).abs() / old_beta.abs()

        if is_zero.sum() > 0:
            diff[is_zero] = (old_beta[is_zero] - new_beta[is_zero]).abs()

        conv_check = diff < tol
        if conv_check.all():
            conv = True
        else:
            conv = False

        return conv

    def _expect_var_y(self, y, w, X, full_beta):
        y_hat = np.dot(X, full_beta)
        y = np.array(y).ravel()
        wr = np.subtract(y, y_hat)
        wr = wr ** 2
        wr = wr.ravel()
        weighted_residual = np.multiply(wr, w)
        var_y = weighted_residual.sum() / self.n
        return var_y

    def _compute_summary(self, r, w):
        vcov_beta = np.linalg.inv(np.dot(r.T, r))
        beta_sd = np.sqrt(np.diag(vcov_beta)) / self.n

        # t score for mean
        t_scores = self.beta / beta_sd
        # two-sided pvalue = Prob(abs(t)>tt)
        p_values = stats.t.sf(np.abs(t_scores), df=self.n - self.k) * 2

        self.beta_sd = beta_sd
        self.p_values = p_values
        self.t_scores = t_scores

        # TODO Calculate weighted residuals and r-sq etc
