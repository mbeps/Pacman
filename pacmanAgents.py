# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1.0
#
# A simple MDP-based agent that uses value iteration.
#
# Extends the basic Agent class from game.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
from copy import deepcopy

class SimpleMDPAgent(Agent):
    """
    An agent that uses value iteration to compute optimal actions
    """
    def __init__(self):
        self.grid = None
        self.utilities = None
        self.rewards = None
        self.width = None 
        self.height = None
        self.discount = 0.9  # Increased discount to make agent more forward-looking
        self.living_reward = -0.04  # Reduced penalty for movement
        self.food_reward = 100  # Increased food reward significantly
        self.iterations = 100  # Maximum number of value iterations
        self.convergence_threshold = 0.01  # Stop when changes are below this
        self.last_score = None
        self.last_food_count = None

    def print_state(self, state):
        """Print the current state of the world for debugging"""
        pacman_pos = api.whereAmI(state)
        food = api.food(state)
        
        print "Current state:"
        print "Pacman position:", pacman_pos
        print "Food locations:", food
        print "Grid representation (P=Pacman, F=Food, #=Wall, .=Empty):"
        
        for y in range(self.height-1, -1, -1):  # Print from top to bottom
            for x in range(self.width):
                if (x,y) == pacman_pos:
                    print "P",
                elif (x,y) in food:
                    print "F",
                elif self.utilities[x][y] is None:
                    print "#",
                else:
                    print ".",
            print
            
        print "Utility values:"
        for y in range(self.height-1, -1, -1):
            for x in range(self.width):
                if self.utilities[x][y] is not None:
                    print "%6.2f" % self.utilities[x][y],
                else:
                    print "   ###",
            print

    def registerInitialState(self, state):
        """Initialize the agent with the game state"""
        # Get grid dimensions from corners
        corners = api.corners(state)
        self.width = max(x for x, y in corners) + 1
        self.height = max(y for x, y in corners) + 1
        
        # Initialize grids for utilities and rewards
        self.utilities = self.create_grid(0.0)
        self.rewards = self.create_grid(self.living_reward)
        
        # Mark walls
        walls = api.walls(state)
        for x, y in walls:
            self.rewards[x][y] = None
            self.utilities[x][y] = None
            
        # Set food rewards
        food = api.food(state)
        for x, y in food:
            self.rewards[x][y] = self.food_reward
        
        self.last_food_count = len(food)
        self.last_score = 0
            
        # Run value iteration
        self.value_iteration()

    def create_grid(self, initial_value):
        """Create a width x height grid with initial_value"""
        return [[initial_value for y in range(self.height)] 
                for x in range(self.width)]

    def value_iteration(self):
        """Perform value iteration to compute utilities for all states"""
        for _ in range(self.iterations):
            # Create a new grid for updated utilities
            new_utilities = self.create_grid(0.0)
            max_change = 0.0
            
            # Update utilities for all states
            for x in range(self.width):
                for y in range(self.height):
                    if self.rewards[x][y] is not None:  # Skip walls
                        # Get maximum expected utility for this state
                        utility = self.compute_state_utility(x, y)
                        new_utilities[x][y] = utility
                        
                        # Track maximum change for convergence check
                        change = abs(utility - self.utilities[x][y])
                        max_change = max(max_change, change)
            
            # Update utilities
            self.utilities = new_utilities
            
            # Check for convergence
            if max_change < self.convergence_threshold:
                break

    def compute_state_utility(self, x, y):
        """Compute utility for a state using the Bellman equation"""
        if self.rewards[x][y] is None:  # Wall
            return None
            
        # Get reward for current state
        R = self.rewards[x][y]
        
        # If it's a terminal state (food), just return the reward
        if R == self.food_reward:
            return R
            
        # Get maximum expected utility over all actions
        max_utility = float("-inf")
        for action in [Directions.NORTH, Directions.SOUTH, 
                      Directions.EAST, Directions.WEST]:
            exp_utility = self.get_expected_utility(x, y, action)
            max_utility = max(max_utility, exp_utility)
            
        # Return utility using Bellman equation
        return R + self.discount * max_utility

    def get_expected_utility(self, x, y, action):
        """Compute expected utility of taking an action in state (x,y)"""
        # Get successor states and their probabilities
        successors = self.get_successor_states(x, y, action)
        
        # Compute expected utility
        exp_utility = 0.0
        for (next_x, next_y), prob in successors.items():
            # Check if successor is valid (not a wall or out of bounds)
            if (0 <= next_x < self.width and
                0 <= next_y < self.height and
                self.utilities[next_x][next_y] is not None):
                exp_utility += prob * self.utilities[next_x][next_y]
            else:
                # If would hit wall or go out of bounds, stay in same place
                exp_utility += prob * self.utilities[x][y]
            
        return exp_utility

    def get_successor_states(self, x, y, action):
        """Return dictionary of successor states and their probabilities"""
        successors = {}
        
        # Get direction vectors for the action and perpendicular movements
        if action == Directions.NORTH:
            intended = (0, 1)
            perpendicular = [(1, 0), (-1, 0)]
        elif action == Directions.SOUTH:
            intended = (0, -1)
            perpendicular = [(1, 0), (-1, 0)]
        elif action == Directions.EAST:
            intended = (1, 0)
            perpendicular = [(0, 1), (0, -1)]
        elif action == Directions.WEST:
            intended = (-1, 0)
            perpendicular = [(0, 1), (0, -1)]
            
        # Add intended direction (0.8 probability)
        next_x = x + intended[0]
        next_y = y + intended[1]
        successors[(next_x, next_y)] = 0.8
        
        # Add perpendicular directions (0.1 probability each)
        for dx, dy in perpendicular:
            next_x = x + dx
            next_y = y + dy
            successors[(next_x, next_y)] = 0.1
            
        return successors

    def getAction(self, state):
        """Get the optimal action using maximum expected utility"""
        # Update rewards based on current food locations
        food = api.food(state)
        current_score = state.getScore()
        
        # Check if state has changed
        if len(food) != self.last_food_count or current_score != self.last_score:
            # Reset non-wall states to living reward
            for x in range(self.width):
                for y in range(self.height):
                    if self.rewards[x][y] is not None:
                        self.rewards[x][y] = self.living_reward
                        
            # Update food rewards
            for x, y in food:
                self.rewards[x][y] = self.food_reward
                
            # Rerun value iteration
            self.value_iteration()
            
            self.last_food_count = len(food)
            self.last_score = current_score
        
        # Print current state for debugging
        self.print_state(state)
        
        # Get current position and legal actions
        x, y = api.whereAmI(state)
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        # Find action with maximum expected utility
        max_utility = float("-inf")
        best_action = None
        
        for action in legal:
            exp_utility = self.get_expected_utility(x, y, action)
            if exp_utility > max_utility:
                max_utility = exp_utility
                best_action = action
                
        # If no legal actions or all have same utility, choose random
        if best_action is None:
            best_action = random.choice(legal)
            
        return api.makeMove(best_action, legal)


def scoreEvaluation(state):
    return state.getScore()
