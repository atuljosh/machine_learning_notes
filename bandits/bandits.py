from __future__ import division
from abc import ABCMeta, abstractmethod
import random
import pandas as pd
import time
import math


class BernoulliArm(object):
    def __init__(self, p):
        self.p = p
        self.count = 0
        self.value_estimate = 0

    def draw_reward(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

    def update_value_estimate(self, reward):
        self.count += 1
        new_value = (((self.count - 1) / self.count) * self.value_estimate) + ((1 / self.count) * reward)
        self.value_estimate = new_value
        return


class BanditAlgorithm(object):
    """
    Abstract bandit algorithm
    """
    __metaclass__ = ABCMeta

    def __init__(self, params=0):
        self.params = params
        self.arms = []

    def initialize_bernoulli_arms(self, means):
        self.arms = map(lambda (mu): BernoulliArm(mu), means)
        return

    @abstractmethod
    def select_arm(self):
        pass

    def test_algorithm(self, means, num_sims, horizon):
        results = []
        for sim in xrange(num_sims):
            self.initialize_bernoulli_arms(means)
            for t in xrange(horizon):
                chosen_arm = self.select_arm()
                reward = self.arms[chosen_arm].draw_reward()
                self.arms[chosen_arm].update_value_estimate(reward)
                result = (self.params, sim, t, chosen_arm, reward)
                results.append(result)

        return results


class EpsilonGreedy(BanditAlgorithm):
    """
    Simple epislon greedy
    """
    def select_arm(self):
        if random.random() > self.params:
            x = [arm.value_estimate for arm in self.arms]
            return x.index(max(x))
        else:
            return random.randrange(len(self.arms))


class SoftMax(BanditAlgorithm):
    """
    Simple softmax / proportional sampling
    with annealing schedule
    """
    def compute_temperature(self):
        if self.params:
            # simple softmax
            temp = self.params
        else:
            # Annealling softmax
            t = sum(i.count for i in self.arms) + 1
            temp = 1 / math.log(t + 0.00000001)
        return temp

    def compute_exponentiated_value_estimates(self, temperature):
        return [math.exp(arm.value_estimate/temperature) for arm in self.arms]

    def compute_proportions(self, exp_values):
        total = sum(exp_values)
        return [arm_value/total for arm_value in exp_values]

    def proportional_sampling(self, proportions):
        rand = random.random()
        cum_prob = 0
        for i in xrange(len(proportions)):
            cum_prob += proportions[i]
            if cum_prob > rand:
                return i
        return len(proportions)-1

    def select_arm(self):
        temperature = self.compute_temperature()
        exp_values = self.compute_exponentiated_value_estimates(temperature=temperature)
        proportions = self.compute_proportions(exp_values)
        chosen_arm_index = self.proportional_sampling(proportions)
        return chosen_arm_index


class UCB1(BanditAlgorithm):
    """
    Simple UCB variant
    """
    def compute_upper_confidence_bounds(self):
        """
            Compute uncertainty estimate for each arm and update value estimate by mean + uncertainty
            uncertainty estimate = math.sqrt((2*math.log(total_counts))/arm.count)
            This is really Chernoff-Hoeffding bound (http://jeremykun.com/2013/04/15/probabilistic-bounds-a-primer/)
            and hence the name upper_confidence_bound
            Simple meaning -
        """
        total_counts = sum(obs.count for obs in self.arms)
        return [arm.value_estimate + math.sqrt((2*math.log(total_counts))/arm.count) for arm in self.arms]

    def select_arm(self):
        # Make sure each arm gets picked at least once
        for idx in xrange(len(self.arms)):
            if self.arms[idx].count == 0:
                return idx

        x = self.compute_upper_confidence_bounds()
        return x.index(max(x))


def run_bandit_algorithm_and_generate_results(algo_name, params=[], sim_nums=1000, times=250):
    random.seed(1)
    means = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    final = []
    params = params if params else [0]

    for param in params:
        if algo_name == 'epsilon_greedy':
            algorithm = EpsilonGreedy(param)
        elif algo_name == 'softmax':
            algorithm = SoftMax(param)
        elif algo_name == 'Annealing_softmax':
            algorithm = SoftMax()
        elif algo_name == 'ucb1':
            algorithm = UCB1()

        results = algorithm.test_algorithm(means, sim_nums, times)
        final.extend(results)

    df = pd.DataFrame(final, columns=['epsilon', 'sim_nums', 'times', 'chosen_arms', 'rewards'])
    grouped_df = df.groupby(['epsilon', 'times']).sum()
    grouped_df['prob_of_best_arm'] = grouped_df['rewards'] / sim_nums

    return grouped_df


if __name__ == '__main__':
    t1 = time.time()

    result_df = run_bandit_algorithm_and_generate_results(algo_name='ucb1', params=[], sim_nums=1000, times=250)

    print result_df

    print "time taken:" + str(time.time()-t1)
