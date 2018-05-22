from model import Model
from bandits import BanditAlgorithm
from games.blackjack.blackjack import BlackJack
from games.gridworld.gridworld import GridWorld
import pandas as pd
import random
import time
import cPickle as pickle
import rl_learning as rl

# Test simple monte-carlo learning for blackjack
def test_training_monte_carlo_for_blackjack(model_class):
    blackjack = BlackJack()
    policy, model = rl.train_reinforcement_learning_strategy(num_sims=5000, game_obs=blackjack, model_class=model_class)

    df = pd.DataFrame(policy).T
    df.columns = ['player_value', 'dealer_value', 'decision', 'score']
    policy_Q_table = df.pivot('player_value', 'dealer_value')['decision']
    print policy_Q_table
    policy_Q_score = df.pivot('player_value', 'dealer_value')['score']
    print policy_Q_score

    # Add ipython notebook 3D ghaph

    # Test policy
    rl.test_policy(blackjack, model)

    return policy, model


# Test TD for blackjack
def test_training_TD_for_blackjack(model_class):
    blackjack = BlackJack()
    policy, model = rl.train_reinforcement_strategy_temporal_difference(epochs=5000, game_obs=blackjack, model_class=model_class)
    df = pd.DataFrame(policy).T
    df.columns = ['player_value', 'dealer_value', 'decision', 'score']
    policy_Q_table = df.pivot('player_value', 'dealer_value')['decision']
    print policy_Q_table
    policy_Q_score = df.pivot('player_value', 'dealer_value')['score']
    print policy_Q_score

    # Add ipython notebook 3D ghaph

    # Test policy
    rl.test_policy(blackjack)

    return policy, model

# Test TD for gridworld
def test_training_TD_for_gridworld(model_class, train=True):
    gridworld = GridWorld()
    if train:
        policy, model = rl.train_reinforcement_strategy_temporal_difference(epochs=50000, game_obs=gridworld, model_class=model_class)
    rl.test_policy(gridworld)

    # Record MSE for each epoch may be?
    # Record % of wins


# Test TD-lambda for gridworld
# TODO For some reason eligibility traces doesn't work as expected
def test_training_TD_lambda_for_gridworld(model_class, train=True):
    gridworld = GridWorld()
    if train:
        policy, model = rl.train_reinforcement_strategy_temporal_difference_eligibility_trace(epochs=2000, game_obs=gridworld, model_class=model_class)
    rl.test_policy(gridworld)



if __name__ == "__main__":
    #policy, model = test_training_monte_carlo_for_blackjack(model_class='lookup_table')
    #policy, model = test_training_monte_carlo_for_blackjack(model_class='vw_python')
    #policy, model = test_training_TD_for_blackjack(model_class='vw_python')
    test_training_TD_for_gridworld(model_class='vw_python', train=True)
    #test_training_TD_lambda_for_gridworld(model_class='vw_python', train=False)