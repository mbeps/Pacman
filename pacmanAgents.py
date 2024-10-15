# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util
import api
from game import Actions


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)


class MyGreedyAgent(Agent):
    def __init__(self):
        self.map = {}
        self.food_value = 10  # Value for spaces with food

    def getAction(self, state):
        # Build or update the map
        self.updateMap(state)
        
        # Get legal actions
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        # Choose the action with maximum expected utility
        return self.getMaxExpectedUtilityAction(state, legal)

    def updateMap(self, state):
        walls = api.walls(state)
        food = api.food(state)
        
        for x in range(state.data.layout.width):
            for y in range(state.data.layout.height):
                if (x, y) not in walls:
                    if (x, y) in food:
                        self.map[(x, y)] = self.food_value
                    else:
                        self.map[(x, y)] = 0

    def getMaxExpectedUtilityAction(self, state, legal):
        pacman_pos = api.whereAmI(state)
        best_action = None
        max_utility = float('-inf')

        for action in legal:
            utility = self.getExpectedUtility(pacman_pos, action)
            if utility > max_utility:
                max_utility = utility
                best_action = action

        return api.makeMove(best_action, legal)

    def getExpectedUtility(self, pos, action):
        x, y = pos
        dx, dy = Actions.directionToVector(action)
        
        # Calculate the positions for the intended move and its left/right alternatives
        intended_pos = (int(x + dx), int(y + dy))
        left_action = Directions.LEFT[action]
        right_action = Directions.RIGHT[action]
        left_pos = (int(x + Actions.directionToVector(left_action)[0]), 
                    int(y + Actions.directionToVector(left_action)[1]))
        right_pos = (int(x + Actions.directionToVector(right_action)[0]), 
                     int(y + Actions.directionToVector(right_action)[1]))

        # Calculate the expected utility
        intended_value = self.map.get(intended_pos, 0)
        left_value = self.map.get(left_pos, 0)
        right_value = self.map.get(right_pos, 0)

        expected_utility = (0.8 * intended_value) + (0.1 * left_value) + (0.1 * right_value)
        return expected_utility

def scoreEvaluation(state):
    return state.getScore()
