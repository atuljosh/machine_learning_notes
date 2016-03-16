import random
import numpy as np
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# Implementation based loosely on: http://outlace.com/Reinforcement-Learning-Part-2/
# each value card has a 1:13 chance of being selected (we don't care about suits for blackjack)

class DecisionState(object):
    def __init__(self):
        self.count = 0
        self.value_estimate = 0  # Without any information assume reward equivalent of DRAW


class BanditAlgorithm(object):
    def __init__(self, params=0):
        self.decision_states = {}
        self.params = params
        self.policy = {}
        self.decisions = None

    def return_decision_reward_tuples(self, information, model):
        table = []
        for decision in model.all_possible_decisions:
            if model.model_class == 'scikit' and hasattr(model.model, 'intercept_'):
                feature_vector, y = model.return_design_matrix(all_decision_states=[(information, decision)])
                reward = model.predict(feature_vector)[0]
                table.append((decision, reward))

            elif model.model_class == 'lookup_table':
                reward = model.predict((information, decision))
                table.append((decision, reward))

        return table

    def return_decision_with_max_reward(self, table):
        table.sort(key=lambda tup: tup[1], reverse=True)
        return table[0]

    def select_decision_given_information(self, information, model=None, algorithm='random'):

        if algorithm == 'epsilon-greedy':
            if random.random() > self.params:

                table = self.return_decision_reward_tuples(information, model)
                # Store policy learned so far
                if table:
                    best_known_decision = self.return_decision_with_max_reward(table)
                    self.policy[information] = [information[0], information[1], best_known_decision[0], best_known_decision[1]]

                else:
                    best_known_decision = (random.choice(model.all_possible_decisions), 0)

            else:
                best_known_decision = (random.choice(model.all_possible_decisions), 0)

            return best_known_decision


class Model(object):
    def __init__(self, params):
        self.model_class = params['class']
        self.model = {}
        self.feature_constructor = None
        self.all_possible_decisions = []
        self.X = []
        self.y = []
        self.buffer = 0

    def initialize(self):
        if self.model_class == 'scikit':
            self.model = SGDRegressor(loss='squared_loss', alpha=0.1, n_iter=10, shuffle=True, eta0=0.0001)
            self.feature_constructor = FeatureHasher(n_features=200, dtype=np.float64, non_negative=False, input_type='dict')

        elif self.model_class == 'lookup':
            self.model = {}

    def clean_buffer(self):
        self.X = []
        self.y = []
        self.buffer = 0

    def return_design_matrix(self, all_decision_states, reward=None):
        if self.model_class == 'lookup_table':
            return all_decision_states, reward

        elif self.model_class == 'scikit':
            X, y = [], []
            for decision_state in all_decision_states:
                information, decision_taken = decision_state
                tr = {}
                tr['-'.join([str(information[1]), decision_taken])] = 1
                tr['-'.join([str(information[0]), decision_taken])] = 1
                tr['-'.join([str(information[0]), str(information[1]), decision_taken])] = 1

                X.append(tr)
                y.extend([reward])
            X = self.feature_constructor.transform(X).toarray()

            return X, y

    def fit(self, X, y):
        if self.model_class == 'scikit':
            # X, y = self.shuffle_data(X, y)
            self.model.partial_fit(X, y)
            print self.model.score(X, y)

        if self.model_class == 'lookup_table':
            for decision_state in X:
                if decision_state not in self.model:
                    for d in self.all_possible_decisions:
                        self.model[(decision_state[0], d)] = DecisionState()

                self.model[decision_state].count += 1
                updated_value = self.model[decision_state].value_estimate + (1.0 / self.model[decision_state].count) * (
                y - self.model[decision_state].value_estimate)
                self.model[decision_state].value_estimate = updated_value

    def predict(self, X):
        if self.model_class == 'scikit':
            return self.model.predict(X)

        if self.model_class == 'lookup_table':
            if X not in self.model:
                for d in self.all_possible_decisions:
                    self.model[(X[0], d)] = DecisionState()
            return self.model[X].value_estimate

    @staticmethod
    def shuffle_data(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]


class BlackJack(object):
    def __init__(self):
        self.player_hand = []
        self.dealer_hand = []
        self.player_value = 0
        self.dealer_value = 0
        self.game_status = ''
        self.decision = ''
        self.state = ()

    def print_game_status(self):
        if self.decision:
            print "player decided to: " + self.decision
        print "player_hand: " + str(self.player_hand) + " with value: " + str(self.player_value)
        print "dealer_hand: " + str(self.dealer_hand) + " with value: " + str(self.dealer_value)
        print self.game_status

    @staticmethod
    def random_card():
        """
        Ace (1), 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack (10), Queen (10), King (10)
        Ace can have value of 1 or 10 based on if other card values < 10
        :return: random card
        """
        card = random.randint(1, 13)
        if card > 10:
            return 10
        return card

    @staticmethod
    def reevaluate_value(hand):
        """
        This is assuming when to use usable_ace
        Ideally an algorithm should also learn this
        along with when to 'hit' and 'stay'
        """
        val = sum(hand)
        if 1 in hand and val <= 11:
            return val + 10
        else:
            return val

    def add_card_to_player(self, card):
        # card = random.choice([2,3,4,9,10])
        # card = random.choice([10])
        self.player_hand.extend([card])
        self.player_value = self.reevaluate_value(self.player_hand)

    def add_card_to_dealer(self, card):
        # card = random.choice([9,8,3,2,4,5,6,1,7])
        # card = random.choice([9, 2])
        self.dealer_hand.extend([card])
        self.dealer_value = self.reevaluate_value(self.dealer_hand)

    def evaluate_game(self, decision=False):
        """
        :return: status
        """
        status = 'in process'
        reward = False
        if not decision:
            if self.player_value == 21:
                if self.dealer_value != 21:
                    status = 'player wins'
                else:
                    status = 'draw'

        if decision == 'stay':
            if (self.dealer_value > 21):
                status = 'dealer busts and player wins'
            elif self.dealer_value == self.player_value:
                status = 'draw'
            elif self.dealer_value < self.player_value:
                status = 'player wins'
            elif self.dealer_value > self.player_value:
                status = 'player loses'

        if decision == 'hit':
            if self.player_value == 21:
                if self.dealer_value != 21:
                    status = 'player wins'
                else:
                    status = 'draw'

            elif self.player_value > 21:
                status = 'player busts and loses'
            elif self.player_value < 21:
                status = 'in process'

        # # win = 5, draw = 2, lose = 1
        if status in ['player wins', 'dealer busts and player wins']:
            reward = 1
        elif status == 'draw':
            reward = 0
        elif status in ['player loses', 'player busts and loses']:
            reward = -1

        return status, reward

    def initiate_game(self):

        # Player gets two cards
        self.add_card_to_player(self.random_card())
        self.add_card_to_player(self.random_card())

        # Let's always hit if card total < 11
        while self.player_value <= 11:
            self.add_card_to_player(self.random_card())

        # Dealer opens a single card
        self.add_card_to_dealer(self.random_card())
        # This card is really hidden from the player
        # self.add_card_to_dealer(self.random_card())

        status, reward = self.evaluate_game()
        self.game_status = status

        return

    def evaluate_dealer(self):
        """
        If player decides to stay:
        the dealer always follows this policy: hit until cards sum to 17 or more, then stay.
        :return:
        """
        while self.dealer_value < 17:
            self.add_card_to_dealer(self.random_card())

    def play(self, decision):
        self.decision = decision
        self.state = ((self.player_value, self.dealer_value), self.decision)
        if decision == 'stay':
            # Evaluate game, dealer plays
            self.evaluate_dealer()
            status, reward = self.evaluate_game(decision)

        if decision == 'hit':
            # If hit, add new card to player's hand
            self.add_card_to_player(self.random_card())
            status, reward = self.evaluate_game(decision)

        self.game_status = status

        return reward

    def complete_one_episode(self, banditAlgorithm, model=None):
        all_decision_states = []
        while self.game_status == 'in process':
            # self.print_game_status()
            information = (self.player_value, self.dealer_value)
            decision, prob = banditAlgorithm.select_decision_given_information(information, algorithm='epsilon-greedy', model=model)

            # Only terminal state returns a valid reward
            reward = self.play(decision)

            all_decision_states.append((information, decision))

        return all_decision_states, reward


def learn_Q_function(all_observed_decision_states, reward, model):
    if model.model_class == 'lookup_table':
        model.fit(all_observed_decision_states, reward)

    elif model.model_class == 'scikit':
        X_new, y_new = model.return_design_matrix(all_observed_decision_states, reward)
        model.X.extend(X_new)
        model.y.extend(y_new)

        if model.buffer == 100:
            model.fit(model.X, model.y)
            model.clean_buffer()

    return model


def train_reinforcement_learning_strategy(num_sims=1, model_class='lookup_table'):

    # Initialize model
    model = Model({'class': model_class})
    banditAlgorithm = BanditAlgorithm(params=0.2)
    model.initialize()
    model.all_possible_decisions = ['hit', 'stay']

    for _ in xrange(num_sims):
        model.buffer += 1

        # Initialize game
        blackjack = BlackJack()
        blackjack.initiate_game()
        if blackjack.game_status != 'in process':
            continue

        all_observed_decision_states, reward = blackjack.complete_one_episode(banditAlgorithm, model)
        model = learn_Q_function(all_observed_decision_states, reward, model)

    return banditAlgorithm.policy, model


if __name__ == "__main__":
    #policy, model = train_reinforcement_learning_strategy(num_sims=500000, model_class='lookup_table')
    policy, model = train_reinforcement_learning_strategy(num_sims=50000, model_class='scikit')
    pd = pd.DataFrame(policy).T
    pd.columns = ['player_value', 'dealer_value', 'decision', 'score']
    pt = pd.pivot('player_value', 'dealer_value')['decision']
    print pt
    pt1 = pd.pivot('player_value', 'dealer_value')['score']
    print pt1
