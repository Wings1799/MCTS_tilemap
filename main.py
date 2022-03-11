import cv2
import keyboard
import numpy as np
from PIL import ImageGrab  # for windows/mac users

from MCTS_tilemap import *
import MCTS_tilemap as mcts
from state_maker import *
import sys
import win32gui
from pynput.keyboard import Key, Controller
import time


def main():
    try:
        threads = int(input('Enter number of threads:'))
    except ValueError:
        print("Not a number")
        return
    
    try:
        simulations = int(input('Enter number of simulations per thread:'))
    except ValueError:
        print("Not a number")
        return

    mcts.THREAD_COUNT = threads
    mcts.NUM_SIMULATIONS = simulations

    np.set_printoptions(threshold=sys.maxsize)
    keyboardLeftRight = Controller()
    keyboardJump = Controller()

    i = 1
    while True:
        state = readScreen()

        if state == None:
            continue
        
        while(not hasNonZero(state[1])):
            print("Player not in image capture, nudging player...")
            keyboardLeftRight.press('d')
            time.sleep(0.05)
            keyboardLeftRight.release('d')
            time.sleep(0.3)
            state = readScreen()
        
        while(not hasNonZero(state[3])):
            print("Goals not in image capture, redoing...")
            state = readScreen()

        root = MonteCarloTreeSearchNode(TilemapState(
            state[0], state[1], state[2], state[3], True))
        selected_node = root.best_action()
        goodChild = selected_node
        actionToDo = goodChild.parent_action

        print(actionToDo)

        if (actionToDo == "LEFT"):
            keyboardLeftRight.press('a')
            time.sleep(0.16)
            keyboardLeftRight.release('a')
        elif (actionToDo == "RIGHT"):
            keyboardLeftRight.press('d')
            time.sleep(0.16)
            keyboardLeftRight.release('d')
        elif (actionToDo == "JUMP_LEFT_SMALL"):
            keyboardLeftRight.press('a')
            keyboardJump.press('w')
            time.sleep(0.225)
            keyboardLeftRight.release('a')
            keyboardJump.release('w')
        elif (actionToDo == "JUMP_RIGHT_SMALL"):
            keyboardLeftRight.press('d')
            keyboardJump.press('w')
            time.sleep(0.225)
            keyboardLeftRight.release('d')
            keyboardJump.release('w')
        elif (actionToDo == "JUMP_LEFT_BIG"):
            keyboardLeftRight.press('a')
            keyboardJump.press('w')
            time.sleep(0.425)
            keyboardLeftRight.release('a')
            keyboardJump.release('w')
        elif (actionToDo == "JUMP_RIGHT_BIG"):
            keyboardLeftRight.press('d')
            keyboardJump.press('w')
            time.sleep(0.425)
            keyboardLeftRight.release('d')
            keyboardJump.release('w')
        time.sleep(0.2)
            
if __name__ == "__main__":
    main()
