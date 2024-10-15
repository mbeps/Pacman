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

class ImprovedGreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None
        self.lastMove = Directions.STOP

    def getAction(self, state):
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        pacman_pos = api.whereAmI(state)
        ghost_positions = api.ghosts(state)

        # Calculate scores for each action
        scored = []
        for action in legal:
            next_pos = self.getNextPosition(pacman_pos, action)
            score = self.evaluationFunction(state)
            
            # Penalize moves that bring Pacman closer to ghosts
            for ghost_pos in ghost_positions:
                distance = util.manhattanDistance(next_pos, ghost_pos)
                if distance <= 2:
                    score -= 500 / (distance + 1)  # Higher penalty for closer ghosts
            
            # Reward moves towards food
            food_list = api.food(state)
            if food_list:
                closest_food = min(food_list, key=lambda food: util.manhattanDistance(next_pos, food))
                score += 10 / (util.manhattanDistance(next_pos, closest_food) + 1)

            scored.append((score, action))

        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        
        # Consider the non-deterministic nature of actions
        if random.random() < 0.8:  # 80% chance of choosing the best action
            chosenAction = random.choice(bestActions)
        else:  # 20% chance of choosing a random legal action
            chosenAction = random.choice(legal)

        return api.makeMove(chosenAction, legal)

    def getNextPosition(self, pos, action):
        x, y = pos
        if action == Directions.NORTH:
            return (x, y + 1)
        elif action == Directions.SOUTH:
            return (x, y - 1)
        elif action == Directions.EAST:
            return (x + 1, y)
        elif action == Directions.WEST:
            return (x - 1, y)
        else:
            return pos



def scoreEvaluation(state):
    return state.getScore()
