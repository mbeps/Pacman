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


class CornerSeekingAgent(object):
    def __init__(self):
        self.corners = None
        self.visited_corners = set()
        self.current_goal = None
        self.path = []
        self.last_position = None
        self.stuck_count = 0

    def getAction(self, state):
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Initialize corners if not done yet
        if not self.corners:
            self.corners = api.corners(state)

        current_position = api.whereAmI(state)

        # Check if we're stuck
        if current_position == self.last_position:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        # If stuck for too long or reached the goal, choose a new corner
        if self.stuck_count > 5 or current_position == self.current_goal:
            self.choose_next_corner(state)
            self.path = []

        # If we don't have a path, calculate one
        if not self.path:
            self.path = self.find_path(current_position, self.current_goal, state)

        # Choose the next move
        if self.path:
            next_move = self.path.pop(0)
        else:
            next_move = random.choice(legal)

        self.last_position = current_position
        return api.makeMove(next_move, legal)

    def choose_next_corner(self, state):
        unvisited = set(self.corners) - self.visited_corners
        if not unvisited:
            self.visited_corners.clear()
            unvisited = set(self.corners)
        current_position = api.whereAmI(state)
        self.current_goal = min(unvisited, key=lambda c: self.manhattan_distance(current_position, c))
        self.visited_corners.add(self.current_goal)

    def find_path(self, start, goal, state):
        frontier = util.Queue()
        frontier.push((start, []))
        explored = set()
        walls = api.walls(state)

        while not frontier.isEmpty():
            (node, path) = frontier.pop()
            if node == goal:
                return path

            if node not in explored:
                explored.add(node)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Up, Right, Down, Left
                    next_node = (node[0] + dx, node[1] + dy)
                    if next_node not in walls:
                        new_path = path + [self.get_direction(node, next_node)]
                        frontier.push((next_node, new_path))
        return []

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_direction(self, from_pos, to_pos):
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


def scoreEvaluation(state):
    return state.getScore()
