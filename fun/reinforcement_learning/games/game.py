from abc import ABCMeta, abstractmethod
import os

class AbstractGame(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.game_status = None
        self.state = None
        self.state_info = None
        self.base_folder_name = None
        self.all_possible_decisions = None

    @abstractmethod
    def initiate_game(self):
        pass

    @abstractmethod
    def complete_one_episode(self):
        pass

    @abstractmethod
    def play(self):
        pass