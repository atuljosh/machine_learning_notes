from itertools import izip
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import add_dummy_feature
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

"""
Things to do:

1) may be comparison with scikit learn's native models - done
2) recheck logistic case

"""


class SimpleSGD(object):
    """
    Very simple sgd
    """

    def __init__(self, params_dict):
        self.lr = params_dict.get('learning_rate', 0.1)
        self.re = params_dict.get('l2_regularization', 0.01)
        self.batch_size = params_dict.get('batch_size', 1)
        self.epochs = params_dict.get('epochs', 10000)
        self.loss_fn = params_dict.get('loss_fn', 'logistic')

        # Regularization params
        self.reg_term = (1 - self.lr * (self.re))

        # Save model
        self.w = None
        self.intercept_ = None
        self.coef_ = None

    @staticmethod
    def sigmoid(hypothesis):
        return 1.0 / (1.0 + np.exp(-hypothesis))

    @staticmethod
    def batch(iterable, n=1):
        # http://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def compute_cost(self, x, y):
        hypothesis = self.compute_hypothesis(x)
        if self.loss_fn == 'logistic':
            log_loss = (y * np.log(hypothesis)) + ((1 - y) * np.log(1 - hypothesis))
            cost = -log_loss.sum()

        elif self.loss_fn == 'squared':
            intercept_penalty = ((self.w[0]) ** 2) * self.re
            coeff_penalty = ((self.w[1:, ]).sum() ** 2) * self.re
            loss = hypothesis - y
            n = len(y)
            cost = ((loss.sum() ** 2) + intercept_penalty + coeff_penalty) / (2 * n)
        return cost

    def compute_gradient(self, x, y, hypothesis):
        loss = hypothesis - y
        if self.loss_fn == 'logistic':
            gradient = np.dot(x.T, loss) / len(x)

        elif self.loss_fn == 'squared':
            gradient = np.dot(x.T, loss) / len(x)

        gradient = gradient.ravel()
        return gradient

    def compute_hypothesis(self, x):
        linear_hypothesis = np.dot(x, self.w)
        if self.loss_fn == 'logistic':
            hypothesis = self.sigmoid(linear_hypothesis)
            return hypothesis

        if self.loss_fn == 'poisson':
            hypothesis = np.log(linear_hypothesis)
            return hypothesis

        elif self.loss_fn == 'squared':
            return linear_hypothesis

    def initialize_weights(self, m):
        # np.random.random((x.columns,))
        self.w = np.random.random((m)) / np.sqrt(m)

    def update_weights(self, gradient):
        # Either do not regularize intercept OR regularize to global mean
        self.w[0] = self.w[0] - self.lr * gradient[0]
        self.w[1:, ] = self.w[1:, ] * self.reg_term - self.lr * gradient[1:, ]

    @staticmethod
    def shuffle_data(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def fit(self, X, y):
        start_time = time.time()

        # Add intercept
        X = add_dummy_feature(X)
        m = len(X[0])
        n = len(X)
        self.initialize_weights(m)
        for _ in xrange(self.epochs):

            # Kind of ugly and slow but all i care about is learning
            X, y = self.shuffle_data(X, y)

            # Create an iterator
            x_iter = self.batch(X, self.batch_size)
            y_iter = self.batch(y, self.batch_size)

            for X_batch, y_batch in izip(x_iter, y_iter):
                hypothesis = self.compute_hypothesis(X_batch)

                # Compute cost for entire dataset
                # cost = self.compute_cost(X, y)
                # print cost

                gradient = self.compute_gradient(X_batch, y_batch, hypothesis)
                self.update_weights(gradient)

        # Save model params for scikit compatibility
        self.intercept_ = self.w[0]
        self.coeff_ = self.w[1:, ]

        print 'elapsed time in training: %f' % (time.time() - start_time)

    def predict(self, X, return_labels=False):
        X = add_dummy_feature(X)  # Add intercept
        hypothesis = self.compute_hypothesis(X)
        if self.loss_fn == 'logistic':
            if return_labels:
                labels = [1 if i > 0.5 else 0 for i in hypothesis]
                return labels
            else:
                return hypothesis

        elif self.loss_fn == 'squared':
            return hypothesis


def synthetic_data(N, loss_fn):
    if loss_fn == 'squared':
        response = np.random.normal(loc=0.1, scale=1, size=N)
    elif loss_fn == 'logistic':
        response = np.random.binomial(n=1, p=0.4, size=N)

    dm_raw = [{
                  'country': np.random.choice(['US', 'UK', 'CA'], p=[0.8, 0.1, 0.1]),
                  'gender': np.random.choice(['M', 'F'], p=[0.3, 0.7]),
                  'bid_type': np.random.choice(['cpc', 'cpm', 'ocpm'], p=[0.2, 0.1, 0.7])}
              for _ in xrange(N)]

    vec = DictVectorizer()
    X = vec.fit_transform(dm_raw).toarray()

    return response, X


def test_model(lm, loss_fn, X_train, y_train, X_test, y_test):
    lm.fit(X_train, y_train)
    y_hat = lm.predict(X_test)

    if loss_fn == 'squared':
        print "mean squared error on test: " + str(mean_squared_error(y_test, y_hat))
    elif loss_fn == 'logistic':
        print "roc_auc_score on test set: " + str(roc_auc_score(y_test, y_hat))

    return y_hat, lm.intercept_, lm.coef_


if __name__ == "__main__":
    loss_fn = 'logistic'
    y, X = synthetic_data(1000, loss_fn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params_dict = {'loss_fn': loss_fn, 'learning_rate': 0.01, 'l2_regularization': 0.001, 'batch_size': 1, 'epochs': 1000}
    our_sgd = SimpleSGD(params_dict)

    print "our model:"
    y_hat, intercept, coeff = test_model(our_sgd, loss_fn, X_train, y_train, X_test, y_test)

    # Scikit learn
    if loss_fn == 'squared':
        scikit_model = LinearRegression()
    elif loss_fn == 'logistic':
        scikit_model = LogisticRegression()

    print "Scikit Learn"
    y_hat, intercept, coeff = test_model(scikit_model, loss_fn, X_train, y_train, X_test, y_test)
