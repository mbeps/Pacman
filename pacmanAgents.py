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

def scoreEvaluation(state):
    return state.getScore()


class GoWestAgent(Agent):
    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        
        # Always try to go WEST if it's a legal move
        if Directions.WEST in legal:
            return Directions.WEST
        
        # If WEST is not possible, choose another direction
        # Here, we prioritize NORTH, then SOUTH, then EAST
        if Directions.NORTH in legal:
            return Directions.NORTH
        elif Directions.SOUTH in legal:
            return Directions.SOUTH
        elif Directions.EAST in legal:
            return Directions.EAST
        
        # If no other direction is possible, stop
        return Directions.STOP


class HungryAgent(Agent):
    def getAction(self, state):
        # Get the current position of Pacman
        pacman_pos = state.getPacmanPosition()
        
        # Get the food grid
        food_grid = state.getFood()
        
        # Find all food positions
        food_positions = food_grid.asList()
        
        # If there's no food left, just stop
        if not food_positions:
            return Directions.STOP
        
        # Find the nearest food
        nearest_food = min(food_positions, key=lambda food: self.manhattan_distance(pacman_pos, food))
        
        # Get legal actions
        legal_actions = state.getLegalPacmanActions()
        
        # Choose the action that gets us closest to the nearest food
        best_action = min(legal_actions, key=lambda action: 
                          self.manhattan_distance(state.generatePacmanSuccessor(action).getPacmanPosition(), nearest_food))
        
        return best_action
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class SurvivalAgent(Agent):
    def getAction(self, state):
        # Get the current position of Pacman
        pacman_pos = state.getPacmanPosition()
        
        # Get the positions of the ghosts
        ghost_states = state.getGhostStates()
        ghost_positions = [ghost.getPosition() for ghost in ghost_states]
        
        # Get legal actions
        legal_actions = state.getLegalPacmanActions()
        
        # Calculate the safety score for each action
        action_scores = {}
        for action in legal_actions:
            next_state = state.generatePacmanSuccessor(action)
            next_pos = next_state.getPacmanPosition()
            
            # Calculate the minimum distance to any ghost
            min_ghost_distance = min(self.manhattan_distance(next_pos, ghost_pos) for ghost_pos in ghost_positions)
            
            # The score is the distance to the nearest ghost (higher is better)
            action_scores[action] = min_ghost_distance
        
        # Choose the action with the highest safety score
        best_action = max(action_scores, key=action_scores.get)
        
        return best_action
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])