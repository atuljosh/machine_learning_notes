__author__ = 'atul'

import random
from itertools import izip
import numpy as np
import patsy as ptsy
from sklearn.metrics import mean_squared_error
import pandas as pd
import time


class sgd:
    """
    Very simple sgd
    """

    def __init__(self, params_dict):
        self.lr = params_dict.get('learning_rate', 0.1)
        self.re = params_dict.get('l2_regularization', 0.01)
        self.batch_size = params_dict.get('batch_size', 1)
        self.epochs = params_dict.get('epochs', 10000)
        self.task = params_dict.get('task', 'classification')

        # Regularization params
        self.reg_term = (1 - self.lr * (self.re))

        # Save model
        self.w = None

    @staticmethod
    def batch(iterable, n=1):
        # http://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def compute_loss(self, hypothesis, y):
        if self.task == 'classification':
            pass

        if self.task == 'regression':
            loss = hypothesis - y
        return loss

    def compute_cost(self, loss, n):
        intercept_penalty = ((self.w[0])**2)*self.re
        coeff_penalty = ((self.w[1:,]).sum()**2)*self.re
        cost = ((loss.sum() ** 2) + intercept_penalty + coeff_penalty) / (2 * n)
        return cost

    def compute_gradient(self, x, loss):
        if self.task == 'classification':
            pass

        if self.task == 'regression':
            gradient = np.dot(x.T, loss) / len(x)
            gradient = gradient.ravel()

        return gradient

    def compute_hypothesis(self, x):
        if self.task == 'classification':
            pass

        if self.task == 'regression':
            hypothesis = np.dot(x, self.w)

        return hypothesis

    def initialize_weights(self, m):
        # np.random.random((x.columns,))
        self.w = np.ones(m, order='C')

    def update_weights(self, gradient):
        self.w[0] = self.w[0] * self.reg_term - self.lr * gradient[0]
        self.w[1:, ] = self.w[1:, ] * self.reg_term - self.lr * gradient[1:, ]

    @staticmethod
    def shuffle_data(X, y):
        combined = zip(X, y)
        random.shuffle(combined)
        X[:], y[:] = zip(*combined)
        return X, y

    def fit(self, X, y):
        start_time = time.time()
        m = len(X[0])
        n = len(X)
        self.initialize_weights(m)
        for _ in xrange(self.epochs):
            # Kind of ugly randomness, but all i care about is learning
            X, y = self.shuffle_data(X, y)

            # Create an iterator
            x_iter = self.batch(X, self.batch_size)
            y_iter = self.batch(y, self.batch_size)

            for X_batch, y_batch in izip(x_iter, y_iter):
                hypothesis = self.compute_hypothesis(X_batch)
                loss = self.compute_loss(hypothesis, y_batch)
                cost = self.compute_cost(loss, n)
                gradient = self.compute_gradient(X_batch, loss)
                self.update_weights(gradient)
                #print cost

        print 'elapsed time in training: %f' % (time.time() - start_time)

    def predict(self, X):
        y_hat = np.dot(X, self.w)
        return y_hat


def new_data(N, pairwise_interactions=False):
    d = {'ctr': np.random.uniform(low=0, high=0.05, size=N),
         'country': [np.random.choice(['US', 'UK', 'CA']) for _ in range(N)],
         'gender': [np.random.choice(['m', 'f']) for _ in range(N)],
         'bid_type': [np.random.choice(['cpa', 'cpc', 'cpm', 'ocpm']) for _ in range(N)],
         'other': [np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']) for _ in range(N)]}

    df = pd.DataFrame(d)

    if pairwise_interactions:
        out = ptsy.dmatrices(
            'ctr ~ country + gender + bid_type + other + country:gender + country:bid_type + country:other + gender:bid_type + gender:other + bid_type:other',
            data=df, return_type='dataframe')

    else:
        out = ptsy.dmatrices('ctr ~ country + gender + bid_type + other', data=df, return_type='dataframe')

    return out


if __name__ == "__main__":
    y, X = new_data(1000)
    y = y.values.ravel()
    X = X.values
    params_dict = {'task': 'regression', 'learning_rate': 0.01, 'l2_regularization': 0.01, 'batch_size': 100, 'epochs': 1000}
    sgd = sgd(params_dict)
    sgd.fit(X, y)

    y_hat = sgd.predict(X)

    print mean_squared_error(y, y_hat)
