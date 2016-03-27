from model import Model
from bandits import BanditAlgorithm
from games.blackjack.blackjack import BlackJack
import pandas as pd
import random


# TODO : 1) Abstract out design matrix for game
# TODO : 2) Implement Q learning and SARSA
# TODO:  3) Start thinking about gridworld

import cProfile
from line_profiler import LineProfiler


def do_cprofile(func):
    """
    Profile as explained here:
    https://zapier.com/engineering/profiling-python-boss/
    """
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func


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

def learn_Q_function(all_observed_decision_states, reward, model):
    # TODO We need to implement experience replay here instead of
    if model.model_class == 'lookup_table':
        model.fit(all_observed_decision_states, reward)

    elif model.model_class == 'scikit' or model.model_class == 'vw':
        for decision_state in all_observed_decision_states:
            X_new, y_new = model.return_design_matrix(decision_state, reward)
            model.X.append(X_new)
            model.y.append(y_new)

        if model.buffer == 1000:
            model.fit(model.X, model.y)

            # TODO Instead of killing entire buffer we can keep a few and kill only the subset
            model.clean_buffer()

    return model


def train_reinforcement_learning_strategy(num_sims=1, game_obs='blackjack', model_class='lookup_table'):
    # Initialize model
    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    banditAlgorithm = BanditAlgorithm(params=0.1)
    model.initialize()

    model.all_possible_decisions = game_obs.all_possible_decisions

    for _ in xrange(num_sims):
        model.buffer += 1

        # Initialize game
        game_obs.initiate_game()
        if game_obs.game_status != 'in process':
            continue

        all_observed_decision_states, reward = game_obs.complete_one_episode(banditAlgorithm, model)
        model = learn_Q_function(all_observed_decision_states, reward, model)

    return banditAlgorithm.policy, model


# TODO New function for training reinforcement strategy
# TODO SUPER SLOW - FIND OUT WHAT IS CAUSING AN ISSUE
@do_profile
def train_reinforcement_strategy_temporal_difference(epochs=1, game_obs='blackjack', model_class='lookup_table'):
    # Initialize model

    # TODO y must be really current reward + discount * max future reward

    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    model.initialize()
    epsilon = 0.5
    banditAlgorithm = BanditAlgorithm(params=epsilon)
    policy = {}
    replay = []
    buffer = 100
    batchsize = 50
    gamma = 0.5
    h=0

    model.all_possible_decisions = game_obs.all_possible_decisions

    for _ in xrange(epochs):
        # Initialize game
        game_obs.initiate_game()
        banditAlgorithm.params = epsilon

        # TODO This assumes we have a dumb model when we initialize
        while game_obs.game_status == 'in process':
            model.buffer += 1
            old_state = game_obs.state
            # TODO Finish implement q value update using Bellman equation
            best_known_decision, known_reward = banditAlgorithm.select_decision_given_information(game_obs.information, model, algorithm='epsilon-greedy')
            # Play or make move to get to a new state and see new reward
            reward = game_obs.play(best_known_decision)
            new_state = game_obs.state

            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((old_state, best_known_decision, reward, new_state))

            # We do not train until buffer is full, but after that we train with every single epoch
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (old_state, best_known_decision, reward, new_state)

                #randomly sample our experience replay memory
                # TODO We don't need batchsize for vw, we can just replay the whole memory may be
                minibatch = random.sample(replay, batchsize)

                # TODO Understand the target network part - Now this is the target_network part,
                for memory in minibatch:

                    old_state, action, reward, new_state = memory
                    new_qval_table = banditAlgorithm.return_decision_reward_tuples(new_state, model)
                    if new_qval_table:
                        maxQ = banditAlgorithm.return_decision_with_max_reward(new_qval_table)

                    if game_obs.game_status == 'in process' and new_qval_table: #non-terminal state
                        reward = (reward + (gamma * maxQ[1]))

                    X_new, y_new = model.return_design_matrix(tuple([(old_state, best_known_decision)]), reward)

                    if model.model_class != 'lookup_table':
                        model.X.extend(X_new)
                        model.y.extend(y_new)
                    else:
                        model.fit([X_new], y_new)

                # We are retraining in every single epoch, but with some subset of all samples
                if model.model_class != 'lookup_table':
                    model.fit(model.X, model.y)

                    # TODO Instead of killing entire buffer we can keep a few and kill only the subset
                    model.clean_buffer()

                print("Game #: %s" % (_,))
                # model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
                # state = new_state
            # if reward != -1: #if reached terminal state, update game status
            #     status = 0

            # # TODO information is really existing state of the game
            policy[old_state] = [old_state[0], old_state[1], best_known_decision, known_reward]

            # TODO Check for terminal state
        if epsilon > 0.1:  # decrement epsilon over time
            epsilon -= (1.0 / epochs)

    return policy, model


if __name__ == "__main__":

    # If we want complete game episodes
    blackjack = BlackJack()
    policy, model = train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class='lookup_table')
    #policy, model = train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class='scikit')
    #policy, model = train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class='vw')

    pd = pd.DataFrame(policy).T
    pd.columns = ['player_value', 'dealer_value', 'decision', 'score']
    policy_Q_table = pd.pivot('player_value', 'dealer_value')['decision']
    print policy_Q_table
    policy_Q_score = pd.pivot('player_value', 'dealer_value')['score']
    print policy_Q_score

## *** ## Now temporal difference learning
    # blackjack = BlackJack()
    # policy, model = train_reinforcement_strategy_temporal_difference(epochs=5000, game_obs=blackjack, model_class='lookup_table')
    # #policy, model = train_reinforcement_strategy_temporal_difference(epochs=2000, game_obs=blackjack, model_class='scikit')
    # #policy, model = train_reinforcement_strategy_temporal_difference(epochs=2000, game_obs=blackjack, model_class='vw')
    # pd = pd.DataFrame(policy).T
    # pd.columns = ['player_value', 'dealer_value', 'decision', 'score']
    # policy_Q_table = pd.pivot('player_value', 'dealer_value')['decision']
    # print policy_Q_table
    # policy_Q_score = pd.pivot('player_value', 'dealer_value')['score']
    # print policy_Q_score