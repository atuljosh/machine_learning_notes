import random
import os

import cProfile
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

class DecisionState(object):
    def __init__(self):
        self.count = 0
        self.value_estimate = 0

class BanditAlgorithm(object):
    def __init__(self, params=0):
        self.decision_states = {}
        self.params = params
        self.policy = {}
        self.decisions = None

    # TODO This may belong outside bandit
    #@do_profile
    def return_decision_reward_tuples(self, information, model):
        q_value_table = []
        for decision in model.all_possible_decisions:
            if model.if_model_exists():
                all_decision_states = tuple([(information, decision)])
                feature_vector, y = model.return_design_matrix(all_decision_states)
                reward = model.predict(feature_vector)
                q_value_table.append((decision, reward))

        return q_value_table

    # TODO This may belong outside bandit
    def return_decision_with_max_reward(self, q_value_table):
        q_value_table.sort(key=lambda tup: tup[1], reverse=True)
        return q_value_table[0]

    def select_decision_given_information(self, information, model=None, algorithm='random'):

        if algorithm == 'epsilon-greedy':
            if random.random() > self.params:

                q_value_table = self.return_decision_reward_tuples(information, model)
                # Store policy learned so far
                if q_value_table:
                    best_known_decision = self.return_decision_with_max_reward(q_value_table)
                    self.policy[information] = [information[0], information[1], best_known_decision[0], best_known_decision[1]]

                else:
                    best_known_decision = (random.choice(model.all_possible_decisions), 0)

            else:
                best_known_decision = (random.choice(model.all_possible_decisions), 0)

            return best_known_decision