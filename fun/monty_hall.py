from __future__ import division
import random


def test():
    choices = ['goat', 'goat', 'bingo']

    # player picks one randomly
    player_pick = choices.pop(choices.index(random.choice(choices)))

    # now host picks one which is a goat - no need for this
    # host_pick = choices.pop(choices.index('goat'))

    prize_of_switch = 0 if player_pick == 'bingo' else 1
    prize_of_no_switch = 1 if player_pick == 'bingo' else 0

    return prize_of_switch, prize_of_no_switch


def monty_monte_carlo(n):
    """
    Simple intuitive explanation:
    Prob that prize behind door 1 = 1/3
    prob that prize behind door 2 OR 3 = 2/3

    User chooses door 1
    Monty kills door 2 or 3 based on which one has a goat, say 2
    (prob that prize behind door 2 OR 3) AND (prize is not behind door 2)
    i.e. Prob that prize behind the remaning door = 2/3
    """
    prize_of_switch = 0
    prize_of_no_switch = 0

    for _ in xrange(n):
        a, b = test()
        prize_of_switch += a
        prize_of_no_switch += b

    prob_win = prize_of_switch / n
    prob_lose = prize_of_no_switch / n

    print "Total number of trials: " + str(n)
    print "probability of winning if switch: {0}". format(prob_win)
    print "probability of winning if no switch: {0}". format(prob_lose)


if __name__ == "__main__":
    monty_monte_carlo(n=100000)