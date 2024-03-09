import time
from utils.key_codes import *
from utils.input_simulation import PressKey, ReleaseKey


############# movement ##############

def go_forward(time_to_go=0.2):
    PressKey(W)
    time.sleep(time_to_go)
    ReleaseKey(W)

def go_backward(time_to_go=0.2):
    PressKey(S)
    time.sleep(time_to_go)
    ReleaseKey(S)

def go_left(time_to_go=0.2):
    PressKey(A)
    time.sleep(time_to_go)
    ReleaseKey(A)

def go_right(time_to_go=0.2):
    PressKey(D)
    time.sleep(time_to_go)
    ReleaseKey(D)

def jump(time_to_go=0.2):
    PressKey(F)
    time.sleep(time_to_go)
    ReleaseKey(F)

def squat(time_to_go=0.2):
    PressKey(X)
    time.sleep(time_to_go)
    ReleaseKey(X)

############# run ##############

def run_forward(time_to_go=0.5):
    PressKey(W)
    PressKey(SPACE)
    time.sleep(time_to_go)
    ReleaseKey(W)
    ReleaseKey(SPACE)

def run_backward(time_to_go=0.5):
    PressKey(S)
    PressKey(SPACE)
    time.sleep(time_to_go)
    ReleaseKey(S)
    ReleaseKey(SPACE)

def run_left(time_to_go=0.5):
    PressKey(A)
    PressKey(SPACE)
    time.sleep(time_to_go)
    ReleaseKey(A)
    ReleaseKey(SPACE)

def run_right(time_to_go=0.5):
    PressKey(D)
    PressKey(SPACE)
    time.sleep(time_to_go)
    ReleaseKey(D)
    ReleaseKey(SPACE)

def do_nothing(time_to_go=0.1):
    time.sleep(time_to_go)

############# dodge ##############

def dodge_backjump(time_to_go=0.1):
    PressKey(SPACE)
    time.sleep(time_to_go)
    ReleaseKey(SPACE)

def dodge_forward():
    PressKey(W)
    time.sleep(0.01)
    PressKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(W)

def dodge_backward():
    PressKey(S)
    time.sleep(0.01)
    PressKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(S)

def dodge_left():
    PressKey(A)
    time.sleep(0.01)
    PressKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(A)

def dodge_right():
    PressKey(D)
    time.sleep(0.01)
    PressKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(SPACE)
    time.sleep(0.01)
    ReleaseKey(D)


############# attack ##############
    
def light_attack(time_to_go=0.01):
    PressKey(J)
    time.sleep(time_to_go)
    ReleaseKey(J)

def heavy_attack(time_to_go=0.01):
    PressKey(K)
    time.sleep(time_to_go)
    ReleaseKey(K)
    time.sleep(0.5)

def special_attack(time_to_go=0.01):
    #TODO: change into single key
    PressKey(I)
    time.sleep(time_to_go)
    ReleaseKey(I)
    time.sleep(1)

def jump_light_attack(time_to_go=0.1):
    PressKey(F)
    time.sleep(time_to_go)
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)
    time.sleep(0.1)
    ReleaseKey(F)

def jump_heavy_attack(time_to_go=0.1):
    PressKey(F)
    time.sleep(time_to_go)
    PressKey(K)
    time.sleep(0.2)
    ReleaseKey(K)
    time.sleep(0.2)
    ReleaseKey(F)

def defend_on(time_to_go=0.4):
    PressKey(L)

def defend_off(time_to_go=0.4):
    ReleaseKey(L)

def defend(time_to_go=0.25):
    PressKey(L)
    time.sleep(time_to_go)
    ReleaseKey(L)


############# other ##############

def focus(time_to_go=0.1):
    PressKey(Q)
    time.sleep(time_to_go)
    ReleaseKey(Q)

def escape():
    PressKey(esc)
    time.sleep(0.4)
    ReleaseKey(esc)

def interact(time_to_go=0.4):
    PressKey(E)
    time.sleep(time_to_go)
    ReleaseKey(E)

def left_key(time_to_go=0.4):
    PressKey(left)
    time.sleep(time_to_go)
    ReleaseKey(left)

def right_key(time_to_go=0.4):
    PressKey(right)
    time.sleep(time_to_go)
    ReleaseKey(right)

def up_key(time_to_go=0.4):
    PressKey(up)
    time.sleep(time_to_go)
    ReleaseKey(up)

def down_key(time_to_go=0.4):
    PressKey(down)
    time.sleep(time_to_go)
    ReleaseKey(down)
    