from model import Model
from bandits import BanditAlgorithm
from games.blackjack.blackjack import BlackJack
from games.gridworld.gridworld import GridWorld
import pandas as pd
import random
import time


# TODO : 2) Implement Q learning and SARSA
# TODO:  3) Implement and see how gridworld works

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

    elif model.model_class == 'scikit' or model.model_class == 'vw' or model.model_class == 'vw_python':
        for decision_state in all_observed_decision_states:
            X_new, y_new = model.return_design_matrix(decision_state, reward)
            model.X.append(X_new)
            model.y.append(y_new)

        if model.buffer == 1000:
            model.fit(model.X, model.y)

            # TODO Instead of killing entire buffer we can keep a few and kill only the subset
            model.clean_buffer()

    return model

#@do_profile
def train_reinforcement_learning_strategy(num_sims=1, game_obs='blackjack', model_class='lookup_table'):
    start_time = time.time()
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

    model.finish()
    elapsed_time = int(time.time() - start_time)
    print ": took time:" + str(elapsed_time)
    return banditAlgorithm.policy, model


# TODO New function for training reinforcement strategy
# TODO SUPER SLOW - FIND OUT WHAT IS CAUSING AN ISSUE
#@do_profile
def train_reinforcement_strategy_temporal_difference(epochs=1, game_obs='blackjack', model_class='lookup_table',algo='q_learning' ):
    # Initialize model

    start_time = time.time()

    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    model.initialize()
    epsilon = 0.2
    banditAlgorithm = BanditAlgorithm(params=epsilon)
    replay = []
    buffer = 500
    batchsize = 100
    gamma = 0.5
    h=0

    model.all_possible_decisions = game_obs.all_possible_decisions

    for _ in xrange(epochs):
        # Initialize game
        game_obs.initiate_game()
        banditAlgorithm.params = epsilon
        move = 0

        # TODO This assumes we have a dumb model when we initialize
        while game_obs.game_status == 'in process':
            move += 1

            # Let's start new game if after 10 moves game doesn't end
            if move > 10:
                break

            model.buffer += 1
            old_state = game_obs.state

            # TODO Finish implement q value update using Bellman equation
            best_known_decision, known_reward = banditAlgorithm.select_decision_given_state(game_obs.state, model,
                                                                                            algorithm='epsilon-greedy')
            # Play or make move to get to a new state and see reward
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

                for memory in minibatch:

                    # TODO Rename these vars.. they are muddying the waters
                    old_state_er, action_er, reward_er, new_state_er = memory

                    if game_obs.game_status == 'in process' and model.exists: #non-terminal state
                        # Get q values for the new state, and then choose best action (a single step temporal difference q learning)
                        # Get value estimate for that best action and update EXISTING reward
                        if algo == 'q_learning':
                            result = banditAlgorithm.return_action_based_on_greedy_policy(new_state_er, model)
                            max_reward = result[1]
                        elif algo == 'sarsa':
                            result = banditAlgorithm.select_decision_given_state(new_state_er, model,
                                                                                            algorithm='epsilon-greedy')
                            max_reward = game_obs.play(result[0])

                        if result:
                            reward_er = (reward_er + (gamma * max_reward))

                    X_new, y_new = model.return_design_matrix((old_state_er, action_er), reward_er)

                    if model.model_class != 'lookup_table':
                        model.X.append(X_new)
                        model.y.append(y_new)
                    else:
                        model.fit([X_new], y_new)

                # We are retraining in every single epoch, but with some subset of all samples
                if model.model_class != 'lookup_table':
                    model.fit(model.X, model.y)

                    # TODO Instead of killing entire buffer we can keep a few and kill only the subset
                    model.clean_buffer()

                print("Game #: %s" % (_,))

            # TODO Check for terminal state
        if epsilon > 0.1:  # decrement epsilon over time
            epsilon -= (1.0 / epochs)

    model.finish()
    elapsed_time = int(time.time() - start_time)
    print ": took time:" + str(elapsed_time)

    return banditAlgorithm.policy, model


def train_reinforcement_strategy_temporal_difference_without_ex_replay(epochs=1, game_obs='blackjack', model_class='lookup_table',algo='q_learning' ):
    # Initialize model

    start_time = time.time()

    model = Model({'class': model_class, 'base_folder_name': game_obs.base_folder_name})
    model.initialize()
    epsilon = 0.2
    banditAlgorithm = BanditAlgorithm(params=epsilon)
    gamma = 0.9
    h=0

    model.all_possible_decisions = game_obs.all_possible_decisions

    for _ in xrange(epochs):
        # Initialize game
        game_obs.initiate_game()
        banditAlgorithm.params = epsilon
        move = 0

        # TODO This assumes we have a dumb model when we initialize
        while game_obs.game_status == 'in process':
            move += 1

            # Let's start new game if after 10 moves game doesn't end
            if move > 10:
                break

            model.buffer += 1
            old_state = game_obs.state

            # TODO Finish implement q value update using Bellman equation
            best_known_decision, known_reward = banditAlgorithm.select_decision_given_state(game_obs.state, model,
                                                                                            algorithm='epsilon-greedy')
            # Play or make move to get to a new state and see reward
            reward = game_obs.play(best_known_decision)
            new_state = game_obs.state

            if game_obs.game_status == 'in process' and model.exists: #non-terminal state
                # Get q values for the new state, and then choose best action (a single step temporal difference q learning)
                # Get value estimate for that best action and update EXISTING reward
                if algo == 'q_learning':
                    result = banditAlgorithm.return_action_based_on_greedy_policy(new_state, model)
                    max_reward = result[1]
                # elif algo == 'sarsa':
                #     result = banditAlgorithm.select_decision_given_state(new_state_er, model,
                #                                                                     algorithm='epsilon-greedy')
                #     max_reward = game_obs.play(result[0])

                if result:
                    reward = (reward + (gamma * max_reward))

            X_new, y_new = model.return_design_matrix((old_state, best_known_decision), reward)

            if model.model_class != 'lookup_table':
                model.X.append(X_new)
                model.y.append(y_new)
            else:
                model.fit([X_new], y_new)

            # We are retraining in every single epoch, but with some subset of all samples
            if model.model_class != 'lookup_table' and model.buffer == 100:
                model.fit(model.X, model.y)
                model.buffer = 0
                #model.clean_buffer()

            print("Game #: %s" % (_,))

            # TODO Check for terminal state
        if epsilon > 0.1:  # decrement epsilon over time
            epsilon -= (1.0 / epochs)

    model.finish()
    elapsed_time = int(time.time() - start_time)
    print ": took time:" + str(elapsed_time)

    return banditAlgorithm.policy, model



def test_policy(game_obs, model):
    print "---------- Testing policy:-----------"
    banditAlgorithm = BanditAlgorithm(params=0.1)
    game_obs.initiate_game()
    print "Initial state:"
    print game_obs.state
    move = 1
    if model.model_class == 'vw_python':
        from vowpal_wabbit import pyvw
        model.model = pyvw.vw("--quiet -i {0}".format(model.model_path))
    while game_obs.game_status == 'in process':
        new_qval_table = banditAlgorithm.return_decision_reward_tuples(game_obs.state, model)
        best_action, value_estimate = banditAlgorithm.return_decision_with_max_reward(new_qval_table)
        print('Move #: %s; Taking action: %s' % (move, best_action))
        reward = game_obs.play(best_action)
        print game_obs.state
        if game_obs.game_status != 'in process': print "Summary: " + game_obs.game_status + " :Player Reward: " + str(reward)
        move += 1

        if move > 10:
            print "Too many moves"
            break

if __name__ == "__main__":

    # If we want complete game episodes
    # blackjack = BlackJack()
    # #policy, model = train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class='lookup_table')
    # #policy, model = train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class='scikit')
    # #policy, model = train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class='vw')
    # policy, model = train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class='vw_python')
    #
    # pd = pd.DataFrame(policy).T
    # pd.columns = ['player_value', 'dealer_value', 'decision', 'score']
    # policy_Q_table = pd.pivot('player_value', 'dealer_value')['decision']
    # print policy_Q_table
    # policy_Q_score = pd.pivot('player_value', 'dealer_value')['score']
    # print policy_Q_score

## *** ## Now temporal difference learning
    # blackjack = BlackJack()
    gridworld = GridWorld()
    #policy, model = train_reinforcement_strategy_temporal_difference(epochs=50000, game_obs=blackjack, model_class='lookup_table')
    #policy, model = train_reinforcement_strategy_temporal_difference(epochs=500, game_obs=blackjack, model_class='scikit')
    #policy, model = train_reinforcement_strategy_temporal_difference(epochs=2000, game_obs=blackjack, model_class='vw')
    #policy, model = train_reinforcement_strategy_temporal_difference(epochs=5000, game_obs=blackjack, model_class='vw_python')

    policy, model = train_reinforcement_strategy_temporal_difference(epochs=500, game_obs=gridworld, model_class='vw_python')
    #policy, model = train_reinforcement_strategy_temporal_difference_without_ex_replay(epochs=500, game_obs=gridworld, model_class='vw_python')

    # pd = pd.DataFrame(policy).T
    # pd.columns = ['player_value', 'dealer_value', 'decision', 'score']
    # policy_Q_table = pd.pivot('player_value', 'dealer_value')['decision']
    # print policy_Q_table
    # policy_Q_score = pd.pivot('player_value', 'dealer_value')['score']
    # print policy_Q_score
    #
    #test_policy(blackjack, model)
    test_policy(gridworld, model)