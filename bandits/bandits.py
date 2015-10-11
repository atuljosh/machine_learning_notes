import random
import numpy as np
import pandas as pd


def ind_max(x):
    return x.index(max(x))


class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for _ in xrange(n_arms)]
        self.values = [0.0 for _ in xrange(n_arms)]
        return

    def select_arm(self):
        if random.random() > self.epsilon:
            return ind_max(self.values)
        else:
            return random.randrange( len( self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Online running weighted average
        new_value = (((n - 1) / float(n)) * value) + ((1 / float(n)) * reward)
        self.values[chosen_arm] = new_value
        return


class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


def test_algorithm(algorithm, arms, num_sims, horizon):
    chosen_arms = [0.0 for _ in xrange(num_sims * horizon)]
    rewards = [0.0 for _ in xrange(num_sims * horizon)]
    cumulative_rewards = [0.0 for _ in xrange(num_sims * horizon)]
    sim_nums = [0.0 for _ in xrange(num_sims * horizon)]
    times = [0.0 for _ in xrange(num_sims * horizon)]
    for sim in xrange(num_sims):
        sim += 1
        algorithm.initialize(len(arms))
        for t in range(horizon):
            t += 1
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t
            chosen_arm = algorithm.select_arm()
            chosen_arms[index] = chosen_arm
            reward = arms[chosen_arms[index]].draw()
            rewards[index] = reward
            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algorithm.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


if __name__ == '__main__':
    random.seed(1)
    means = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    n_arms = len(means)
    random.shuffle(means)
    arms = map(lambda (mu): BernoulliArm(mu), means)
    print(" Best arm is " + str(ind_max(means)))
    final = []
    for epsilon in [0.1]:
        algorithm = EpsilonGreedy(epsilon, [], [])
        algorithm.initialize(n_arms)
        results = test_algorithm(algorithm, arms, 300, 25)
        eps = [epsilon] * len(results[0])
        results.insert(0, eps)
        results = zip(*results)
        final.extend(results)

    df = pd.DataFrame(final, columns=['epsilon', 'sim_nums', 'times', 'chosen_arms', 'rewards', 'cumulative_rewards'])
    grouped_df = df.groupby(['epsilon', 'times'])
    grouped_df['prob_of_best_arm'] = grouped_df['rewards'] / 3000
    print grouped_df

