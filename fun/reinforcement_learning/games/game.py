from abc import ABCMeta, abstractmethod
import os

class AbstractGame(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.game_status = None
        self.state = None
        self.base_folder_name = None
        self.all_possible_decisions = None

    @abstractmethod
    def initiate_game(self):
        """
        This should set the initial state
        """
        pass

    @abstractmethod
    def complete_one_episode(self):
        """
        Used for Monte-Carlo learning
        """
        pass

    @abstractmethod
    def play(self):
        """
        This should update state after interacting with the
        environment. Mostly used for temporal difference learning
        :return: reward
        """
        pass