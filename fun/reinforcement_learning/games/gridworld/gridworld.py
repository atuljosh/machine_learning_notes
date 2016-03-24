import random
from abc import ABCMeta, abstractmethod
from collections import namedtuple
#from .. import game
import os
import numpy as np
import pandas as pd

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

class GridWorld(AbstractGame):

    def __init__(self):
        self.base_folder_name = os.path.dirname(os.path.realpath(__file__))
        self.all_possible_decisions = ['up', 'down', 'left', 'right']
        self.coordinates = namedtuple('coordinates', ['x', 'y'])
        self.all_used_coordinates = {'x': set(), 'y': set()}
        self.state_info = None
        self.game_status = None
        self.state = None

    def random_unsed_coordinates(self, a, b):
        # TODO Fix this
        xa = np.random.randint(a, b)
        while xa not in self.all_used_coordinates['x']:
            ya = np.random.randint(a, b)
            while ya not in self.all_used_coordinates['y']:
                self.all_used_coordinates['x'].add(xa)
                self.all_used_coordinates['y'].add(ya)
                return xa, ya

    def initiate_game(self):
        """
        What exactly we know about the game beforehand?
         - all possible actions we can take
         - and image (we don't know player, wall, pit or goal)
         - game's internal environment returns reward based on the movement and that's how we should learn the rules
         - so our design matrix should only be set of pixels (or image representation) AND decision taken
         - all we know is : there are 4 things on the screen.. lets use sparse representation
        """
        x, y = 0,2 #self.random_unsed_coordinates(0, 4)
        self.player_info = self.coordinates(x, y)
        x, y = 1,3 #self.random_unsed_coordinates(0, 4)
        self.wall_info = self.coordinates(x, y)
        x, y = 2,3#self.random_unsed_coordinates(0, 4)
        self.pit_info = self.coordinates(x, y)
        x, y = 3,3 #self.random_unsed_coordinates(0, 4)
        self.win_info = self.coordinates(x, y)
        self.state_info = (self.player_info, self.wall_info, self.pit_info, self.win_info)

    def display_grid(self):
        grid = np.zeros((4,4), dtype='<U2')

        grid[self.player_info.x, self.player_info.y] = 'P'
        grid[self.wall_info.x, self.wall_info.y] = 'W'
        if self.player_info != self.pit_info:
            grid[self.pit_info.x, self.pit_info.y] = '-'

        if self.player_info != self.win_info:
            grid[self.win_info.x, self.win_info.y] = '+'

        print pd.DataFrame(grid)


    def complete_one_episode(self):
        pass

    def play(self, action):
        if action == 'left':
            new_loc = self.coordinates(self.player_info.x, self.player_info.y-1)
            if new_loc != self.wall_info and new_loc.y >= 0:
                self.player_info = new_loc

        elif action == 'right':
            new_loc = self.coordinates(self.player_info.x, self.player_info.y+1)
            if new_loc != self.wall_info and new_loc.y <= 3:
                self.player_info = new_loc

        elif action == 'up':
            new_loc = self.coordinates(self.player_info.x-1, self.player_info.y)
            if new_loc != self.wall_info and new_loc.x >= 0:
                self.player_info = new_loc

        elif action == 'down':
            new_loc = self.coordinates(self.player_info.x+1, self.player_info.y)
            if new_loc != self.wall_info and new_loc.x <= 3:
                self.player_info = new_loc

    def get_reward(self):
        if self.player_info == self.pit_info:
            return -10
        elif self.player_info == self.win_info:
            return 10
        else:
            return -1



if __name__ == "__main__":
    gridworld = GridWorld()
    gridworld.initiate_game()
    print gridworld.player_info
    gridworld.display_grid()
    gridworld.play('down')
    print gridworld.player_info
    gridworld.display_grid()
    gridworld.play('down')
    print gridworld.player_info
    print gridworld.display_grid()
    gridworld.play('down')
    print gridworld.player_info
    print gridworld.display_grid()
    gridworld.play('right')
    print gridworld.player_info
    print gridworld.display_grid()

    print gridworld.get_reward()