import time
from utils.get_window import *
import utils.utils_actions as actions

def setup_game(name="ELDEN RING"):
    # select the window
    partial_window_title = name
    select_window_by_partial_title(partial_window_title)
    time.sleep(1)

    # get into game
    actions.escape()
    time.sleep(1)

    # start the game
    start_game()


def restart_game():
    time.sleep(18)
    actions.interact()
    time.sleep(15)

    start_game()


def start_game():
    actions.go_forward(10)
    time.sleep(0.2)
    actions.go_right(1)
    time.sleep(0.2)
    actions.go_forward(1.5)
    time.sleep(0.2)
    # get into the cave
    actions.interact()
    time.sleep(0.2)
    actions.left_key(0.05)
    time.sleep(0.2)
    actions.interact()
    time.sleep(5)
    # activate the battle
    actions.go_forward(4)
    time.sleep(0.2)
    actions.focus()
    time.sleep(1)
