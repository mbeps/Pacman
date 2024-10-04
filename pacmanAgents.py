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

class ForagingSurvivalAgent(Agent):
    def __init__(self):
        self.lastMove = Directions.STOP
        self.state = "foraging"
        self.goal = None
        self.path = []

    def getAction(self, state):
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        pacman_pos = api.whereAmI(state)
        ghost_positions = api.ghosts(state)

        # Check for nearby ghosts
        for ghost in ghost_positions:
            if util.manhattanDistance(pacman_pos, ghost) <= 3:
                self.state = "survival"
                break
        else:
            self.state = "foraging"

        if self.state == "survival":
            return self.survive(state, legal, ghost_positions)
        else:
            return self.forage(state, legal)

    def survive(self, state, legal, ghost_positions):
        pacman_pos = api.whereAmI(state)
        best_move = None
        max_distance = -1

        for action in legal:
            next_pos = self.getNextPosition(pacman_pos, action)
            min_ghost_distance = min(util.manhattanDistance(next_pos, ghost) for ghost in ghost_positions)
            
            if min_ghost_distance > max_distance:
                max_distance = min_ghost_distance
                best_move = action

        return api.makeMove(best_move, legal)

    def forage(self, state, legal):
        food_list = api.food(state)
        pacman_pos = api.whereAmI(state)

        if not self.goal or self.goal not in food_list:
            if food_list:
                self.goal = min(food_list, key=lambda food: util.manhattanDistance(pacman_pos, food))
                self.path = self.bfs(state, pacman_pos, self.goal)
            else:
                return api.makeMove(random.choice(legal), legal)

        if self.path:
            next_move = self.path.pop(0)
            return api.makeMove(next_move, legal)
        else:
            return api.makeMove(random.choice(legal), legal)

    def bfs(self, state, start, goal):
        queue = util.Queue()
        queue.push((start, []))
        visited = set()
        walls = api.walls(state)

        while not queue.isEmpty():
            (node, path) = queue.pop()
            if node == goal:
                return path

            if node not in visited:
                visited.add(node)
                for next_node in self.getNeighbors(node, walls):
                    new_path = path + [self.getDirection(node, next_node)]
                    queue.push((next_node, new_path))

        return []

    def getNeighbors(self, pos, walls):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_pos = (x + dx, y + dy)
            if next_pos not in walls:
                neighbors.append(next_pos)
        return neighbors

    def getDirection(self, from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH

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
