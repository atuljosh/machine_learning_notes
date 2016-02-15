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

    print "-------- Simple Monte Carlo Simulation -------------------"
    print "Total number of trials: " + str(n)
    print "probability of winning if switch: {0}". format(prob_win)
    print "probability of winning if no switch: {0}". format(prob_lose)

def monty_bayes_explanation(number_of_choices):
    """
    Bayes explanation of monty game
    p(h/d) = p(h)*p(d/h) / p(d)
    """
    #Setup
    choices = ['goat', 'goat', 'bingo']

    goat_choices = [i for i, ltr in enumerate(choices) if ltr == 'goat']

    #step one - player picks one randomly
    player_pick = goat_choices.index(random.choice(goat_choices))

    # now host has to to pick a goat
    host_pick = [i for i in goat_choices if i != player_pick][0]

    # Now compute probablity of switching vs probability of keeping the same option
    # p(not_switch_and_win/host_pick_goat) = p(not_switch_and_win) * p(host_pick_goat/not_switch) / p(host_pick_goat)
    # p(switch_and_win/host_pick_goat) = p(switch_and_win) * p(host_pick_goat/switch) / p(host_pick_goat)

    # Prior p(win) = 1/3
    # p(host_pick_goat/win_with_player_pick) = 1/2 (host can pick either of the doors)
    # p(player_pick_and_win) = 1/3
    # p(host_pick_goat) = 1/2

    # p(host_pick_goat/win_with_switch) = 1 (host has to pick the one with goat)
    # p(switch_and_win) = 1/3
    # p(host_pick_goat) = 1/2

    print "\n-------------- Now the Bayesian Approach -------------------"
    print "Prior probability of player winning:" + str(1/3)
    print "player pick door: " + str(player_pick)

    print "host opens door: " + str(host_pick)

    print """
    p(host_pick_goat) = p(player_pick_and_win)*p(host_pick_goat/win_with_player_pick) +
                         p(switch_and_win)*p(host_pick_goat/win_with_switch) +
                         p(host_pick_goat/win_with_host_pick)
                      = 1/3 * 1/2 + 1/3 * 1 + 0 = 1/2
    """

    print "\nBased on Baye's rule:"
    print "First calculate:----------"
    print "p(not_switch_and_win/host_pick_goat) = p(not_switch_and_win) * p(host_pick_goat/not_switch_and_win) / p(host_pick_goat)"

    print "\np(not_switch_and_win) (OR prior probability): " + str(1/3)
    print "probability of p(host opens {0}/player pick {1} and player wins): {2}".format(host_pick, player_pick, 1/2)
    print "p(host_pick_goat: " + str(1/2)
    print "p(not_switch_and_win/host_pick_goat) = (1/3 * 1/2) / (1/2) = 1/3"

    print "\nNow Calculate:---------"
    print "p(switch_and_win/host_pick_goat) = p(switch_and_win) * p(host_pick_goat/switch_and_win) / p(host_pick_goat)"
    print "probability of p(host opens {0}/player pick {1} and player doesn't win): {2}".format(host_pick, player_pick, 1)
    print "p(switch_and_win/host_pick_goat) = (1/3 * 1) / (1/2) = 2/3"

    print "\nConfirmed Monte Carlo Simulation was correct!!!"

if __name__ == "__main__":
    monty_monte_carlo(n=100000)
    monty_bayes_explanation(number_of_choices=3)


