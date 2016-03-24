from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import bandits as bandit
from subprocess import Popen, PIPE, STDOUT
import os
from tempfile import NamedTemporaryFile

from line_profiler import LineProfiler
def do_profile(func):
    def profiled_func(*args, **kwargs):
        try:
            profiler = LineProfiler()
            profiler.add_function(func)
            profiler.enable_by_count()
            return func(*args, **kwargs)
        finally:
            profiler.print_stats()
    return profiled_func


class Model(object):
    def __init__(self, params):
        self.model_class = params['class']
        self.model = {}
        self.feature_constructor = None
        self.all_possible_decisions = []
        self.X = []
        self.y = []
        self.buffer = 0
        self.base_folder_name = params['base_folder_name']
        self.design_matrix_cache = {}

    def initialize(self):
        if self.model_class == 'scikit':
            self.model = SGDRegressor(loss='squared_loss', alpha=0.1, n_iter=10, shuffle=True, eta0=0.0001)
            self.feature_constructor = FeatureHasher(n_features=200, dtype=np.float64, non_negative=False, input_type='dict')

        elif self.model_class == 'lookup':
            self.model = {}

        elif self.model_class == 'vw':
            self.model = None

            self.model_path = self.base_folder_name + "/model.vw"
            self.cache_path = self.base_folder_name + "/temp.cache"
            self.f1 = open(self.base_folder_name + "/train.vw", 'a')

            self.train_vw_cmd = ['/usr/local/bin/vw', '--save_resume', '--holdout_off', '-c', '--cache_file',self.cache_path,
                                 '-f', self.model_path, '--passes', '20', '--loss_function', 'squared']
            self.train_vw_resume_cmd = ['/usr/local/bin/vw', '--save_resume',
                                 '-i', self.model_path, '-f', self.model_path]
            #self.remove_vw_files()

    def remove_vw_files(self):
        if os.path.isfile(self.cache_path): os.remove(self.cache_path)
        if os.path.isfile(self.f1): os.remove(self.f1)
        if os.path.isfile(self.model_path): os.remove(self.model_path)

    def if_model_exists(self):
        exists = False
        if self.model_class == 'lookup_table':
            if self.model:
                exists = True

        elif self.model_class == 'scikit':
            if hasattr(self.model, 'intercept_'):
                exists = True

        elif self.model_class == 'vw':
            if os.path.isfile(self.model_path):
                exists = True
        return exists

    def clean_buffer(self):
        self.X = []
        self.y = []
        self.buffer = 0

    # TODO Store design matrix in cache so we don't have to compute it all the time
    #@do_profile
    def return_design_matrix(self, all_decision_states, reward=None):
        """
        Design matrix can simply return catesian product of information and decision
        For now all categorical features
        """
        # TODO Kill game specific features
        if self.model_class == 'lookup_table':
            return all_decision_states[0], reward

        elif self.model_class == 'scikit':
            # if all_decision_states in self.design_matrix_cache:
            #     X, y = self.design_matrix_cache[all_decision_states]
            # else:
            X, y = [], []
            for decision_state in all_decision_states:
                information, decision_taken = decision_state
                all_features = ['-'.join([i, str(j), decision_taken]) for i,j in zip(information._fields, information)]
                all_features_with_interaction = all_features + ['_'.join(all_features)]
                tr = {fea_value: 1 for fea_value in all_features_with_interaction}
                X.append(tr)
                y.extend([reward])
            X = self.feature_constructor.transform(X).toarray()
                # self.design_matrix_cache[all_decision_states] = (X, y)
            return X, y

        elif self.model_class == 'vw':
            X, y = [], []
            for decision_state in all_decision_states:
                fv = " "
                information, decision_taken = decision_state
                all_features = ['-'.join([i, str(j), decision_taken]) for i,j in zip(information._fields, information)]
                all_features_with_interaction = all_features + ['_'.join(all_features)]
                input = " ".join(all_features_with_interaction)

                if reward:
                    output = str(reward) + " " + '-'.join([str(information[0]), str(information[1]), decision_taken])
                    fv = output + " |" + input
                else:
                    fv = " |" + input
                X.append(fv)
                y.extend([reward])
            X = '\n'.join(X) + '\n'
            if reward:
                self.f1.write(X)
            return [X], y

    def fit(self, X, y):
        if self.model_class == 'scikit':
            # X, y = self.shuffle_data(X, y)
            self.model.partial_fit(X, y)
            print self.model.score(X, y)

        elif self.model_class == 'lookup_table':
            for decision_state in X:
                if decision_state not in self.model:
                    for d in self.all_possible_decisions:
                        self.model[(decision_state[0], d)] = bandit.DecisionState()

                self.model[decision_state].count += 1
                updated_value = self.model[decision_state].value_estimate + (1.0 / self.model[decision_state].count) * (
                y - self.model[decision_state].value_estimate)
                self.model[decision_state].value_estimate = updated_value

        elif self.model_class == 'vw':
            # if model file exists do --save resume
            # http://stackoverflow.com/questions/13835055/python-subprocess-check-output-much-slower-then-call
            if os.path.isfile(self.model_path):
                with NamedTemporaryFile() as f:
                    p = Popen(self.train_vw_resume_cmd, stdout=f, stdin=PIPE, stderr=STDOUT)
                    tr = '\n'.join(X)
                    res=p.communicate(tr)
                    f.seek(0)
                    res = f.read()

            # else train a new model
            else:
                p = Popen(self.train_vw_cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                tr = '\n'.join(X)
                res=p.communicate(tr)
                print res

    def predict(self, test):
        if self.model_class == 'scikit':
            return self.model.predict(test)[0]

        elif self.model_class == 'lookup_table':
            if test not in self.model:
                for d in self.all_possible_decisions:
                    self.model[(test[0], d)] = bandit.DecisionState()
            return self.model[test].value_estimate

        elif self.model_class == 'vw':
            with NamedTemporaryFile() as f:
                cmd = ['/usr/local/bin/vw', '-t', '-i', self.model_path, '-p', '/dev/stdout', '--quiet']
                p = Popen(cmd, stdout=f, stdin=PIPE, stderr=STDOUT)
                tr = '\n'.join(test)
                res = p.communicate(tr)
                f.seek(0)
                res = f.readline().strip()
            return float(res)


    @staticmethod
    def shuffle_data(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]