import numpy as np
from collections import defaultdict
from copy import *
from math import *
import random
import multiprocessing

NUM_SIMULATIONS = 25
THREAD_COUNT = 8
verbose = False
noJumping = False
weightedDirections = False

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state

        while not current_rollout_state.is_game_over():

            possible_moves = current_rollout_state.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):

        choices_weights = [(c.q() / c.n()) + c_param *
                           np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        processes = THREAD_COUNT

        pool = multiprocessing.Pool(processes)

        tasks = [self] * processes

        finishedTasks = pool.map(branchNode, tasks)
        pool.close()
        pool.join()

        combineNodes(self, finishedTasks)

        return self.best_child(c_param=0.)
    
    def tree_branching(self):
        simulation_no = NUM_SIMULATIONS
        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        if (verbose):
            print("Done branching")
        return self

def branchNode(node):
    node.tree_branching()
    if (verbose):
            print("Done branching helper function")
    return node


def combineNodes(targetNode:MonteCarloTreeSearchNode, nodeList:list[MonteCarloTreeSearchNode]):
    for node in nodeList:
        # targetNode.children += node.children
        for child in node.children:
            index = childExists(targetNode, child)
            if index < 0:
                targetNode.children.append(child)
            else:
                targetNode.children[index]._number_of_visits += child._number_of_visits
                targetNode.children[index]._results[1] += child._results[1]
                targetNode.children[index]._results[-1] += child._results[-1]

        targetNode._number_of_visits += node._number_of_visits
        targetNode._results[1] += node._results[1]
        targetNode._results[-1] += node._results[-1]

def childExists(targetNode:MonteCarloTreeSearchNode, childNode:MonteCarloTreeSearchNode):
    index = -1
    i = 0
    for child in targetNode.children:
        if child.state.playerMap == childNode.state.playerMap:
            index = i
        i += 1
    return index

class TilemapState():
    def __init__(self, tileMap, playerMap, hazardMap, goalMap, justStarted=False):
        self.tileMap = tileMap
        self.playerMap = playerMap
        self.hazardMap = hazardMap
        self.goalMap = goalMap
        self.justStarted = justStarted
        return

    def get_legal_actions(self):
        row, column = self.find_player()
        # print(row, " ", column)

        if (noJumping):
            possibleActions = ["LEFT", "RIGHT"]
        else:
            possibleActions = ["LEFT", "RIGHT", "JUMP_RIGHT_SMALL",
                               "JUMP_RIGHT_BIG", "JUMP_LEFT_SMALL", "JUMP_LEFT_BIG"]

        if column == 0:
            removeFromList(possibleActions, "LEFT")
        if column < 2:
            removeFromList(possibleActions, "JUMP_LEFT_SMALL")
        if column < 4:
            removeFromList(possibleActions, "JUMP_LEFT_BIG")
        if column == 31:
            removeFromList(possibleActions, "RIGHT")
        if column > 29:
            removeFromList(possibleActions, "JUMP_RIGHT_SMALL")
        if column > 27:
            removeFromList(possibleActions, "JUMP_RIGHT_BIG")
        if row < 3:
            removeFromList(possibleActions, "JUMP_RIGHT_SMALL")
            removeFromList(possibleActions, "JUMP_RIGHT_BIG")
            removeFromList(possibleActions, "JUMP_LEFT_SMALL")
            removeFromList(possibleActions, "JUMP_LEFT_BIG")

        hazards = self.find_hazards()
        for hazard in hazards:
            for columnOffset in [0, 1, 2]:
                if hazard[1] == column + columnOffset and row - hazard[0] <= 3:
                    removeFromList(possibleActions, "JUMP_RIGHT_BIG")
            for columnOffset in [0, 1]:
                if hazard[1] == column + columnOffset and row - hazard[0] <= 2:
                    removeFromList(possibleActions, "JUMP_RIGHT_SMALL")
            for columnOffset in [0, -1, -2]:
                if hazard[1] == column + columnOffset and row - hazard[0] <= 3:
                    removeFromList(possibleActions, "JUMP_LEFT_BIG")
            for columnOffset in [0, -1]:
                if hazard[1] == column + columnOffset and row - hazard[0] <= 2:
                    removeFromList(possibleActions, "JUMP_LEFT_SMALL")
        # goals = self.find_goals()
        tiles = self.find_tiles()

        isAirborne = True

        for block in tiles:
            if block[1] == column - 1 and block[0] == row:
                removeFromList(possibleActions, "LEFT")
            if block[1] == column + 1 and block[0] == row:
                removeFromList(possibleActions, "RIGHT")
            if block[1] == column and block[0] == row - 1:
                removeFromList(possibleActions, "JUMP_RIGHT_SMALL")
                removeFromList(possibleActions, "JUMP_RIGHT_BIG")
                removeFromList(possibleActions, "JUMP_LEFT_SMALL")
                removeFromList(possibleActions, "JUMP_LEFT_BIG")
            if block[0] == row + 1:
                isAirborne = False

        if isAirborne:
            for removeMe in ["JUMP_RIGHT_SMALL", "JUMP_RIGHT_BIG", "JUMP_LEFT_SMALL", "JUMP_LEFT_BIG"]:
                if removeMe in possibleActions:
                    possibleActions.remove(removeMe)

        if weightedDirections:
            noRight = False
            noLeft = False
            bestGoal = self.getBestGoal()
            if column > bestGoal[1]:
                weightedBool = [True] * 70 + [False] * 30
                noRight = random.choice(weightedBool)
            elif column < bestGoal[1]:
                weightedBool = [True] * 70 + [False] * 30
                noLeft = random.choice(weightedBool)

            if noRight:
                removeFromList(possibleActions, "JUMP_RIGHT_SMALL")
                removeFromList(possibleActions, "JUMP_RIGHT_BIG")
                removeFromList(possibleActions, "RIGHT")
            if noLeft:
                removeFromList(possibleActions, "JUMP_LEFT_SMALL")
                removeFromList(possibleActions, "JUMP_LEFT_BIG")
                removeFromList(possibleActions, "LEFT")

        if (verbose):
            print("YOUR LEGAL ACTIONS", possibleActions)

        if len(possibleActions) == 0:
            possibleActions.append("NOTHING")

        return possibleActions

    def find_player(self):
        row = -1
        column = -1

        for i in self.playerMap:
            column = -1
            row += 1
            for j in i:
                column += 1
                if j == 1:
                    if (verbose):
                        print("PLAYER: ", row, " ", column)
                    return [row, column]
        row = -1
        column = -1

        if (row == -1 and column == -1):
            raise Exception("CAN'T FIND PLAYER. YOU SHOULDN'T SEE THIS")

        if (verbose):
            print("PLAYER: ", row, " ", column)
        return [row, column]

    def find_goals(self):
        row = -1
        allGoals = []

        for i in self.goalMap:
            column = -1
            row += 1
            for j in i:
                column += 1
                if j == 1:
                    allGoals.append([row, column])
                    if (verbose):
                        print("GOAL: ", row, " ", column)

        return allGoals

    def find_hazards(self):
        row = -1
        allHazards = []

        for i in self.hazardMap:
            column = -1
            row += 1
            for j in i:
                column += 1
                if j == 1:
                    allHazards.append([row, column])
                    if (verbose):
                        print("HAZARD: ", row, " ", column)

        return allHazards

    def find_tiles(self):
        row = -1
        allTiles = []

        for i in self.tileMap:
            column = -1
            row += 1
            for j in i:
                column += 1
                if j == 1:
                    allTiles.append([row, column])
                    # print("TILE: ", row, " ", column)

        return allTiles

    def is_game_over(self):
        return (self.game_result() == 1 or self.game_result() == -1)

    def game_result(self):
        row, column = self.find_player()
        hazards = self.find_hazards()
        goals = self.find_goals()

        for hazard in hazards:
            if row == hazard[0] and column == hazard[1]:
                return -1
            elif row + 1 == hazard[0] and column == hazard[1]:
                return -1

        for goal in goals:
            if row == goal[0] and column == goal[1]:
                return 1
            elif row + 1 == goal[0] and column == goal[1]:
                return 1

        if (column == 0 or column == 31 or row == 31) and not self.justStarted:
            return -1

        return 0

    def getBestGoal(self):
        goals = self.find_goals()
        row, column = self.find_player()
        hazards = self.find_hazards()
        tiles = self.find_tiles()
        bestGoalWeight = 9999999
        bestGoal = None

        for goal in goals:
            distance = self.distanceToGoal(goal)
            multiplier = 10
            rowTilesBlocking = 0
            columnTilesBlocking = 0

            for blocks in [tiles, hazards]:
                if row > goal[0]:
                    for block in blocks:
                        if block[0] < row and block[1] == column:
                            rowTilesBlocking += 1
                elif row < goal[0]:
                    for block in blocks:
                        if block[0] > row and block[1] == column:
                            rowTilesBlocking += 1
                if column > goal[1]:
                    for block in blocks:
                        if block[1] < column and block[0] == row:
                            columnTilesBlocking += 1
                elif column < goal[1]:
                    for block in blocks:
                        if block[1] > column and block[0] == row:
                            columnTilesBlocking += 1

            if rowTilesBlocking == 0:
                multiplier *= 0.5
            if columnTilesBlocking == 0:
                multiplier *= 0.5

            goalDistance = distance * (rowTilesBlocking + 1) * (columnTilesBlocking + 1) * multiplier
            if(verbose):
                print("GOALWEIGHT VALUE: ", goalDistance)
            if goalDistance < bestGoalWeight:
                bestGoalWeight = goalDistance
                bestGoal = goal
        
        return bestGoal

    def distanceToGoal(self, goal):
        row, column = self.find_player()
        xDist = abs(row - goal[0])
        yDist = abs(column - goal[1])
        hypotenuse = hypot(xDist, yDist)

        return abs(hypotenuse)

    def move(self, action):
        if (verbose):
            print("TAKING ACTION: ", action)
        newPlayerMap = deepcopy(self.playerMap)
        row, column = self.find_player()

        newPlayerMap[row][column] = 0
        newRow = row
        newColumn = column

        if (action == "LEFT"):
            newColumn = column - 1
        if (action == "RIGHT"):
            newColumn = column + 1
        if (action == "JUMP_RIGHT_SMALL"):
            newRow = row - 2
            newColumn = column + 2
        if (action == "JUMP_RIGHT_BIG"):
            newRow = row - 2
            newColumn = column + 4
        if (action == "JUMP_LEFT_SMALL"):
            newRow = row - 2
            newColumn = column - 2
        if (action == "JUMP_LEFT_BIG"):
            newRow = row - 2
            newColumn = column - 4

        allTiles = [self.find_tiles(), self.find_goals(), self.find_hazards()]

        isAirborne = True

        for blocks in allTiles:
            for block in blocks:
                if block[1] == newColumn and block[0] == newRow + 1:
                    isAirborne = False

        if (isAirborne):
            closestFoothold = 32
            for tiles in allTiles:
                for tile in tiles:
                    if tile[1] != newColumn:
                        continue
                    if tile[0] < newRow:
                        continue
                    if tile[0] < closestFoothold:
                        closestFoothold = tile[0]
            newRow = closestFoothold - 1

        newPlayerMap[newRow][newColumn] = 1

        return TilemapState(self.tileMap, newPlayerMap, self.hazardMap, self.goalMap, False)


def removeFromList(someList: list, object):
    if object in someList:
        someList.remove(object)
