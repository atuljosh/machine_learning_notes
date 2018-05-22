from core_glm import GLM
import numpy as np

# TODO Test case when don't want to regularize intercept with global mean

class LinearModel(GLM):

    def fit(self, X, y, weights):
        self._set_variables(X)

        # Get initial weights
        weight_count = self._set_w(weights)

        # Get initial values for beta
        intercept_start, full_beta = self._get_starting_values(y)

        if self.loss == 'squared':
            var_y = np.var(y).values

        # We never learn variance for glms
        else:
            var_y = 1
            y_raw = y
            # Recalculate y and w for glms (so for gaussian, this step is not needed)
            fitted_values = np.dot(X, full_beta)
            y = self._get_pseudo_y(fitted_values, y_raw)
            weights = self._get_pseudo_w(fitted_values, weight_count)

        if self.prior_mean:
            if self.regularize_intercept and 'Intercept' not in self.prior_mean:
                self.prior_mean['Intercept'] = intercept_start
            x_star = self._get_x_star(X)
            y_star = self._get_y_star(X, y)
        else:
            x_star = X
            y_star = y

        # Get w_star based on prior var
        w_star = self._get_w_diag(var_y, weights)
        q, r = self._get_qr(x_star, w_star)

        old_beta = full_beta
        # EM / ILS for beta and prior variance
        for i in xrange(self.max_iter):
            # M-step - get most likely betas given variance(s)
            beta = self._get_new_beta(q, r, y_star, w_star)

            # E-step - need expectation of variance(s) given the betas
            # Also IRLS procedure for glms
            if self.loss != 'squared':
                fitted_values = np.dot(X, beta)
                y = self._get_pseudo_y(fitted_values, y_raw)
                weights = self._get_pseudo_w(fitted_values, weight_count)
                y_star = self._get_y_star(X, y)
                var_y = 1
            else:
                var_y = self._expect_var_y(y, weights, X, full_beta)

            w_star = self._get_w_diag(var_y, weights)
            q, r = self._get_qr(x_star, w_star)

            # Check for convergence
            if self._if_converged(old_beta, beta, tol=0.00001):
                break

            old_beta = beta

        # Save final beta
        final_beta = beta
        self.beta = final_beta

        # TODO Decorate final summary
        self._compute_summary(r, weight_count)

        return final_beta


    def predict(self, X):
        return np.dot(X, self.beta)

    def transform_model(self):
        pass

    def summary(self, r):
        pass



