from __future__ import division
from abc import ABCMeta, abstractmethod
import random
import pandas as pd
import time
import math
import numpy as np


class BanditArm(object):
    """
    Arm is just one possible control or decision configuration available to US
    This can be a:
        - bernoulli arm (just 1/0 reward like a click or conversion)
        - binomial or poisson arm (prob of click or arrival rate)
        - normal or lognormal (value per click or value per conversion)

    """
    def __init__(self, true_params=None, prior_params=None):

        # True distrbution params
        if true_params:
            self.true_distribution = true_params.get('distribution', 'bernoulli')
            self.p = true_params.get('p', 0.5) # Binom, bernoulli
            self.n = true_params.get('n', 1)   # binom
            self.mu = true_params.get('mu', 0.5) # normal
            self.lamb = true_params.get('lamb', 0.5) # poisson
        else:
            print 'CANNOT PROCEED, Need known arm distributions to proceed'

        # Arm statistics / distribution parameter estimates
        self.trials = 0
        self.sample_mean = 0
        self.upper_confidence_bound = 0
        self.sample_variance = 0
        self.quantity_used_for_arm_selection = 0

        # Prior beta distribution parameters, both 1 means uniform distribution
        if prior_params:
            self.prior_alpha = prior_params.get('alpha', 1) # Beta (for beta-binomial posterior)
            self.prior_beta = prior_params.get('beta', 1)   # Beta (for beta-binomial posterior)
            self.prior_mu = prior_params.get('mu', 1) # For normal-normal posterior
        else:
            self.prior_alpha, self.prior_beta, self.prior_mu = 1, 1, 1

    def update_trial_count_for_arm(self):
        self.trials += 1

    def return_reward_for_trial(self):
        """
        Observed reward will be based on true distribution of an arm
        e.g. bernoulli, binomial, poisson, normal, lognormal etc
        """
        if self.true_distribution == 'bernoulli':
            return np.random.binomial(n=1, p=self.p)

    def update_sample_mean_mle(self, reward):
        """
        Compute simple Maximum likelihood estimate of reward
        For bernoulli, binomial, poisson and gaussian mle estimate of paramter is just sample mean: number_of_successes / number of trials
        Note: we are calculating running avg, (n-1)/n * old_estimate + 1/n * latest_reward
        :param reward:
        :return:
        """
        new_value = (((self.trials - 1) / self.trials) * self.sample_mean) + ((1 / self.trials) * reward)
        self.sample_mean = new_value
        return

    def update_upper_confidence_bound(self, total_trials):
        """
            Compute uncertainty estimate for each arm and update value estimate by mean + uncertainty
            uncertainty estimate = math.sqrt((2*math.log(total_counts))/arm.count)
            This is really Chernoff-Hoeffding bound (http://jeremykun.com/2013/04/15/probabilistic-bounds-a-primer/)
            and hence the name upper_confidence_bound
        """
        self.upper_confidence_bound = self.sample_mean + math.sqrt((2 * math.log(total_trials)) / self.trials)
        return self.upper_confidence_bound

    def update_sample_mean_map(self, reward):
        """
        Compute beta-binomial posterior of reward
        (Beta is conjugate prior of bernoulli/binomial)
        Beta mean = alpha / (alpha+beta)
        :param reward:
        :return:
        """
        if reward > 0:
            self.prior_alpha += 1
        else:
            self.prior_beta += 1
        self.sample_mean = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        return


class BanditPolicy(object):
    """
    Abstract bandit algorithm
    """
    __metaclass__ = ABCMeta

    def __init__(self, params=0):
        self.params = params
        self.arms = []
        self.total_trials = 0

    def initialize_bandit_arms(self, known_arm_probabilities):
        self.arms = [BanditArm(true_params={'distribution': 'bernoulli', 'p': prob}) for prob in known_arm_probabilities]
        return

    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def update_arm_sample_mean(self, reward, arm):
        pass

    # @abstractmethod
    def modify_quantity_used_for_arm_selection(self, arm):
        pass

    def test_algorithm(self, means, num_sims, horizon):
        results = []
        for sim in xrange(num_sims):
            self.initialize_bandit_arms(means)
            for t in xrange(horizon):
                self.total_trials += 1
                chosen_arm = self.select_arm()
                reward = chosen_arm.return_reward_for_trial()
                chosen_arm.update_trial_count_for_arm()
                self.update_arm_sample_mean(reward, chosen_arm)
                #self.modify_quantity_used_for_arm_selection(chosen_arm)
                result = (self.params, sim, t, chosen_arm.p, reward)
                results.append(result)

        return results


class EpsilonGreedy(BanditPolicy):
    """
    Simple epislon greedy
    """
    def modify_quantity_used_for_arm_selection(self, arm):
        """
        Epsilon-greedy uses sample_mean for selection without modification
        :param arm:
        :return:
        """
        arm.quantity_used_for_arm_selection = arm.sample_mean
        return

    def select_arm(self):
        if random.random() > self.params:
            x = [arm.sample_mean for arm in self.arms]
            max_idx = x.index(max(x))
            return self.arms[max_idx]
        else:
            return np.random.choice(self.arms)

    def update_arm_sample_mean(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_mle(reward)
        return


class SoftMax(BanditPolicy):
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
            t = self.total_trials + 1
            temp = 1 / math.log(t + 0.00000001)
        return temp

    def compute_exponentiated_value_estimates(self, temperature):
        return [math.exp(arm.sample_mean/temperature) for arm in self.arms]

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
        return self.arms[chosen_arm_index]

    def update_arm_sample_mean(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_mle(reward)
        return


class UCB1(BanditPolicy):
    """
    Simple UCB variant
    """
    def select_arm(self):
        # Make sure each arm gets picked at least once
        for arm in self.arms:
            if arm.trials == 0:
                return arm

        x = [arm.update_upper_confidence_bound(self.total_trials) for arm in self.arms]
        max_idx = x.index(max(x))
        return self.arms[max_idx]

    def update_arm_sample_mean(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_mle(reward)
        return


class ThompsonSampling(BanditPolicy):
    # Start with alpha and beta
    # For each arm, draw arm based on beta(alpha, beta)
    # Choose arm such that, max reward, keep count of success and failuer
    # update beta params for each arm
    # continue
    def select_arm(self):
        """
        Select arm with max estimated map mean
        :return:
        """
        samples = [np.random.beta(arm.prior_alpha, arm.prior_beta) for arm in self.arms]
        max_id = samples.index(max(samples))
        return self.arms[max_id]

    def update_arm_sample_mean(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_map(reward)
        return


def run_bandit_algorithm_and_generate_results(policy_name, params=[], sim_nums=1000, times=250):
    random.seed(1)
    known_arm_probabilities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    final = []
    params = params if params else [0]

    for param in params:
        if policy_name == 'epsilon_greedy':
            policy = EpsilonGreedy(param)
        elif policy_name == 'softmax':
            policy = SoftMax(param)
        elif policy_name == 'annealing_softmax':
            policy = SoftMax()
        elif policy_name == 'ucb1':
            policy = UCB1()
        elif policy_name == 'thompson_sampling':
            policy = ThompsonSampling()

        results = policy.test_algorithm(known_arm_probabilities, sim_nums, times)
        final.extend(results)

    df = pd.DataFrame(final, columns=['epsilon', 'sim_nums', 'times', 'chosen_arms', 'rewards'])
    grouped_df = df.groupby(['epsilon', 'times']).sum()
    grouped_df['prob_of_best_arm'] = grouped_df['rewards'] / sim_nums

    return grouped_df


if __name__ == '__main__':
    t1 = time.time()
    result_df = run_bandit_algorithm_and_generate_results(policy_name='epsilon_greedy', params=[0.1], sim_nums=1000, times=250)
    print result_df
    print "time taken:" + str(time.time()-t1)
