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

        # Prior beta distribution parameters, both 1 means uniform distribution
        if prior_params:
            self.prior_distribution = prior_params.get('distribution', 'beta')
            self.prior_alpha = prior_params.get('alpha', 1) # Beta (for beta-binomial posterior)
            self.prior_beta = prior_params.get('beta', 1)   # Beta (for beta-binomial posterior)
            self.prior_mu = prior_params.get('mu', 1) # For normal-normal posterior
        else:
            self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_distribution = 1, 1, 1, 'beta'

    def update_trial_count_for_arm(self):
        self.trials += 1

    def observe_reward_for_trial(self):
        """
        Observed reward will be based on true distribution of an arm
        e.g. bernoulli, binomial, poisson, normal, lognormal etc
        """
        self.update_trial_count_for_arm()
        if self.true_distribution == 'bernoulli':
            return np.random.binomial(n=1, p=self.p)

    def update_sample_mean_mle(self, reward):
        """
        Compute simple Maximum likelihood estimate of reward
        For bernoulli, binomial, poisson and gaussian mle estimate of paramter is just sample mean: number_of_successes / number of trials
        Note: we are calculating running avg, (n-1)/n * old_estimate + 1/n * latest_reward

        For contexual bandits, sample_mean estimate can be a linear (or whatever) function of features
        Contexual bandits are powerful since features can share information i.e. you don't need to pull arm explicitely for learning its sample mean

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
        if self.true_distribution == 'bernoulli' or self.true_distribution == 'binomial':
            return self.sample_mean + math.sqrt((2 * math.log(total_trials)) / self.trials) if self.trials > 0 else 0

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

    def sample_from_posterior_distribution(self):
        """
        If arm true distribution is bernoulli or binomial
        prior distribution is beta, so sample from beta
        :return:
        """
        if self.true_distribution == 'bernoulli' or self.true_distribution == 'binomial':
            return np.random.beta(self.prior_alpha, self.prior_beta)


class BanditPolicy(object):
    """
    Abstract bandit algorithm
    """
    __metaclass__ = ABCMeta

    def __init__(self, params=0):
        self.params = params
        self.arms = None
        self.total_trials = 0

    def initialize_policy_and_bandit_arms(self, arm_params):
        arm_reward_distribution = arm_params.get('distribution', 'bernoulli')
        self.arms = [BanditArm(true_params={'distribution': arm_reward_distribution, 'p': prob}) for prob in arm_params.get('known_arm_probabilities')]
        self.total_trials = 0
        return

    @abstractmethod
    def select_and_return_best_arm(self):
        pass

    @abstractmethod
    def update_sample_mean_for_selected_arm(self, reward, arm):
        pass

    @abstractmethod
    def return_selection_statistics_for_all_arms(self):
        pass

    def return_idx_for_max_value(self, x):
        return x.index(max(x))

    def test_policy(self, means, num_sims, times):
        results = []
        for sim in xrange(num_sims):
            self.initialize_policy_and_bandit_arms(means)

            for t in xrange(times):
                self.total_trials += 1
                chosen_arm = self.select_and_return_best_arm()
                reward = chosen_arm.observe_reward_for_trial()
                self.update_sample_mean_for_selected_arm(reward, chosen_arm)
                result = (self.params, sim, t, chosen_arm.p, reward)
                results.append(result)

        return results


class EpsilonGreedy(BanditPolicy):
    """
    Simple epislon greedy
    """
    def return_selection_statistics_for_all_arms(self):
        return [arm.sample_mean for arm in self.arms]

    def select_and_return_best_arm(self):
        if random.random() > self.params:
            x = self.return_selection_statistics_for_all_arms()
            max_idx = self.return_idx_for_max_value(x)
            return self.arms[max_idx]
        else:
            return np.random.choice(self.arms)

    def update_sample_mean_for_selected_arm(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_mle(reward)
        return


class SoftMax(BanditPolicy):
    """
    Simple softmax / proportional sampling
    with annealing schedule
    """
    def return_selection_statistics_for_all_arms(self):
        temperature = self.compute_temperature()
        exp_values = self.compute_exponentiated_value_estimates(temperature=temperature)
        proportions = self.compute_proportions(exp_values)
        return proportions

    def compute_temperature(self):
        if self.params:
            # simple softmax
            temp = self.params
        else:
            # Annealling softmax
            temp = 1 / math.log(self.total_trials + 0.00000001)
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

    def select_and_return_best_arm(self):
        proportions = self.return_selection_statistics_for_all_arms()
        chosen_arm_index = self.proportional_sampling(proportions)
        return self.arms[chosen_arm_index]

    def update_sample_mean_for_selected_arm(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_mle(reward)
        return


class UCB1(BanditPolicy):
    """
    Simple UCB variant
    """
    def return_selection_statistics_for_all_arms(self):
        return [arm.update_upper_confidence_bound(self.total_trials) for arm in self.arms]

    def select_and_return_best_arm(self):
        # Make sure each arm gets picked at least once
        for arm in self.arms:
            if arm.trials == 0:
                return arm

        x = self.return_selection_statistics_for_all_arms()
        max_idx = self.return_idx_for_max_value(x)
        return self.arms[max_idx]

    def update_sample_mean_for_selected_arm(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_mle(reward)
        return


class ThompsonSampling(BanditPolicy):
    # Start with alpha and beta
    # For each arm, draw arm based on beta(alpha, beta)
    # Choose arm such that, max reward, keep count of success and failuer
    # update beta params for each arm
    # continue
    def return_selection_statistics_for_all_arms(self):
        return [arm.sample_from_posterior_distribution() for arm in self.arms]

    def select_and_return_best_arm(self):
        """
        Select arm with max estimated map mean
        :return:
        """
        samples = self.return_selection_statistics_for_all_arms()
        max_id = self.return_idx_for_max_value(samples)
        return self.arms[max_id]

    def update_sample_mean_for_selected_arm(self, reward, chosen_arm):
        chosen_arm.update_sample_mean_map(reward)
        return


def run_bandit_algorithm_and_generate_results(policy_name, params=[], sim_nums=1000, times=250):
    random.seed(1)
    arm_params = {'distribution': 'bernoulli', 'known_arm_probabilities': [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]}
    final = []
    params = params if params else [0]
    policy = None

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

        results = policy.test_policy(arm_params, sim_nums, times)
        final.extend(results)

    df = pd.DataFrame(final, columns=['epsilon', 'sim_nums', 'times', 'chosen_arms', 'rewards'])
    grouped_df = df.groupby(['epsilon', 'times']).sum()
    grouped_df['prob_of_best_arm'] = grouped_df['rewards'] / sim_nums

    return grouped_df


if __name__ == '__main__':
    t1 = time.time()
    result_df = run_bandit_algorithm_and_generate_results(policy_name='thompson_sampling', params=[0.1], sim_nums=1000, times=250)
    print result_df
    print "time taken:" + str(time.time()-t1)
