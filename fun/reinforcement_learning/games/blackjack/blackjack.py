import random
from collections import namedtuple
from .. import game
import os

# TODO Resolve for information <===> state.. unneccessary confusion

class BlackJack(game.AbstractGame):

    def __init__(self):
        self.base_folder_name = os.path.dirname(os.path.realpath(__file__))
        self.all_possible_decisions = ['hit', 'stay']

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
        #card = random.choice([10])
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

        self.player_hand = []
        self.dealer_hand = []
        self.player_value = 0
        self.dealer_value = 0
        self.game_status = ''
        self.decision = ''
        self.state = ()
        self.state_info = namedtuple('state_info', ['player_value', 'dealer_value'])
        self.information = None

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
        self.state = self.state_info(self.player_value, self.dealer_value)
        #self.information = self.state_info(self.player_value, self.dealer_value)

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
        if decision == 'stay':
            # Evaluate game, dealer plays
            self.evaluate_dealer()
            status, reward = self.evaluate_game(decision)

        if decision == 'hit':
            # If hit, add new card to player's hand
            self.add_card_to_player(self.random_card())
            status, reward = self.evaluate_game(decision)

        self.game_status = status
        self.state = self.state_info(self.player_value, self.dealer_value)
        #self.information = self.state_info(self.player_value, self.dealer_value)

        return reward

    def complete_one_episode(self, banditAlgorithm, model=None):
        all_decision_states = []
        while self.game_status == 'in process':
            # self.print_game_status()
            state = self.state_info(self.player_value, self.dealer_value)
            decision, prob = banditAlgorithm.select_decision_given_state(state, model=model, algorithm='epsilon-greedy')

            # Only terminal state returns a valid reward
            reward = self.play(decision)

            all_decision_states.append((state, decision))

        all_decision_states_tuple = tuple(all_decision_states)
        return all_decision_states_tuple, reward


