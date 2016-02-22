import random


# Implementation based loosely on: http://outlace.com/Reinforcement-Learning-Part-2/ &
# https://inst.eecs.berkeley.edu/~cs188/sp08/projects/blackjack/blackjack.py
# each value card has a 1:13 chance of being selected (we don't care about suits for blackjack)


class DecisionState(object):
    def __init__(self):
        self.count = 0
        self.value_estimate = 0  # Without any information assume reward equivalent of DRAW

    def update_value_estimate(self, reward):
        """
        Calculate running average of value estimate for a given decision-state
        """
        self.count += 1
        new_value = self.value_estimate + (1 / self.count) * (reward - self.value_estimate)
        self.value_estimate = new_value


class BanditAlgorithm(object):
    def __init__(self, params=0):
        self.decision_states = {}
        self.params = params
        self.policy = {}

    def select_decision_given_information(self, information, algorithm='random'):
        if algorithm == 'random':
            decision = random.choice(['hit', 'stay'])

        if algorithm == 'epsilon-greedy':
            if random.random() > self.params and information in self.decision_states:
                x = [(decision, obs.value_estimate) for decision, obs in self.decision_states[information].iteritems()]
                x.sort(key=lambda tup: tup[1], reverse=True)
                return x[0][0]
            else:
                return random.choice(['hit', 'stay'])

        return decision


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
        self.player_hand.extend([card])
        self.player_value = self.reevaluate_value(self.player_hand)

    def add_card_to_dealer(self, card):
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

    def complete_one_round(self, banditAlgorithm):
        all_decision_states = []
        all_available_decisions_at_any_state = ['hit', 'stay']
        while self.game_status == 'in process':
            # self.print_game_status()
            information = (self.player_value, self.dealer_value)
            decision = banditAlgorithm.select_decision_given_information(information, 'epsilon-greedy')

            # Store policy
            banditAlgorithm.policy[information] = decision

            # Only terminal state returns a valid reward
            reward = self.play(decision)
            all_decision_states.append((information, decision))

        # Store and update value estimates for all decision-state tuples based on the observed reward
        for decision_state in all_decision_states:
            information, decision_taken = decision_state

            if decision_state in banditAlgorithm.decision_states:
                banditAlgorithm.decision_states[decision_state].update_value_estimate(reward)
            else:
                if information not in banditAlgorithm.decision_states:
                    banditAlgorithm.decision_states[information] = {d: DecisionState() for d in all_available_decisions_at_any_state}

                banditAlgorithm.decision_states[information][decision].update_value_estimate(reward)


def train_reinforcement_learning_strategy(num_sims=1):
    banditAlgorithm = BanditAlgorithm()
    for _ in xrange(num_sims):
        blackjack = BlackJack()
        blackjack.initiate_game()
        blackjack.complete_one_round(banditAlgorithm)

    print str(banditAlgorithm.policy)


if __name__ == "__main__":

    banditAlgorithm = BanditAlgorithm(params=0.1)
    for _ in xrange(100000):
        blackjack = BlackJack()
        blackjack.initiate_game()
        blackjack.complete_one_round(banditAlgorithm)

    print banditAlgorithm.policy
    # train_reinforcement_learning_strategy(num_sims=1)
