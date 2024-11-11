# mdpAgents.py
# parsons/20-nov-2017
#
# Version 2.0
#
# An MDP-based agent that handles ghosts using value iteration.

from pacman import Directions
from game import Agent
import api
import random
import game
import util
from copy import deepcopy

class GhostMDPAgent(Agent):
    """
    An agent that uses value iteration to compute optimal actions,
    now with ghost avoidance/chasing capabilities
    """
    def __init__(self):
        # Grid representation
        self.grid = None
        self.utilities = None
        self.rewards = None
        self.width = None 
        self.height = None
        
        # MDP parameters
        self.discount = 0.9
        self.living_reward = -0.04
        self.food_reward = 100
        self.iterations = 100
        self.convergence_threshold = 0.01
        
        # Ghost-related parameters
        self.ghost_reward = -500      # Large negative for ghost locations
        self.ghost_near_reward = -100 # Smaller negative for adjacent squares  
        self.ghost_scared_reward = 200 # Positive reward for scared ghosts
        self.capsule_reward = 100     # Reward for power capsules
        
        # State tracking
        self.last_score = None
        self.last_food_count = None
        self.last_ghost_states = None

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
            
        # Set capsule rewards
        capsules = api.capsules(state)
        for x, y in capsules:
            self.rewards[x][y] = self.capsule_reward
            
        # Initialize ghost states
        self.last_ghost_states = api.ghostStates(state)
        self.last_food_count = len(food)
        self.last_score = 0
            
        # Run initial value iteration
        self.value_iteration()

    def create_grid(self, initial_value):
        """Create a width x height grid with initial_value"""
        return [[initial_value for y in range(self.height)] 
                for x in range(self.width)]

    def value_iteration(self):
        """Perform value iteration to compute utilities for all states"""
        for _ in range(self.iterations):
            new_utilities = self.create_grid(0.0)
            max_change = 0.0
            
            for x in range(self.width):
                for y in range(self.height):
                    if self.rewards[x][y] is not None:
                        utility = self.compute_state_utility(x, y)
                        new_utilities[x][y] = utility
                        change = abs(utility - self.utilities[x][y])
                        max_change = max(max_change, change)
            
            self.utilities = new_utilities
            
            if max_change < self.convergence_threshold:
                break

    def compute_state_utility(self, x, y):
        """Compute utility for a state using the Bellman equation"""
        if self.rewards[x][y] is None:  # Wall
            return None
            
        R = self.rewards[x][y]
        
        # Get maximum expected utility over all actions
        max_utility = float("-inf")
        for action in [Directions.NORTH, Directions.SOUTH, 
                      Directions.EAST, Directions.WEST]:
            exp_utility = self.get_expected_utility(x, y, action)
            max_utility = max(max_utility, exp_utility)
            
        return R + self.discount * max_utility

    def get_expected_utility(self, x, y, action):
        """Compute expected utility of taking an action in state (x,y)"""
        successors = self.get_successor_states(x, y, action)
        
        exp_utility = 0.0
        for (next_x, next_y), prob in successors.items():
            # Check if successor is valid
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
        
        # Get direction vectors
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

    def add_adjacent_rewards(self, x, y, reward):
        """Add rewards to squares adjacent to (x,y)"""
        adjacents = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for adj_x, adj_y in adjacents:
            if (0 <= adj_x < self.width and 
                0 <= adj_y < self.height and 
                self.rewards[adj_x][adj_y] is not None):
                # Only update if it would make the reward more extreme
                if reward < 0:
                    self.rewards[adj_x][adj_y] = min(
                        self.rewards[adj_x][adj_y], 
                        reward
                    )
                else:
                    self.rewards[adj_x][adj_y] = max(
                        self.rewards[adj_x][adj_y], 
                        reward
                    )

    def getAction(self, state):
        """Get the optimal action using maximum expected utility"""
        # Get current state information
        ghost_states = api.ghostStates(state)
        food = api.food(state)
        capsules = api.capsules(state)
        current_score = state.getScore()
        
        # Check if state has changed
        state_changed = (len(food) != self.last_food_count or 
                        current_score != self.last_score or
                        ghost_states != self.last_ghost_states)
                        
        if state_changed:
            # Reset non-wall states to living reward
            for x in range(self.width):
                for y in range(self.height):
                    if self.rewards[x][y] is not None:
                        self.rewards[x][y] = self.living_reward
            
            # Update food rewards
            for x, y in food:
                self.rewards[x][y] = self.food_reward
                
            # Update capsule rewards
            for x, y in capsules:
                self.rewards[x][y] = self.capsule_reward
                
            # Update ghost rewards based on state
            for (ghost_x, ghost_y), scared in ghost_states:
                ghost_x = int(ghost_x)
                ghost_y = int(ghost_y)
                
                if scared:
                    # If ghost is scared, it's a positive reward
                    self.rewards[ghost_x][ghost_y] = self.ghost_scared_reward
                    # Make adjacent squares slightly positive
                    self.add_adjacent_rewards(ghost_x, ghost_y, self.ghost_scared_reward * 0.5)
                else:
                    # If ghost is dangerous, it's a negative reward
                    self.rewards[ghost_x][ghost_y] = self.ghost_reward
                    # Make adjacent squares negative but less so
                    self.add_adjacent_rewards(ghost_x, ghost_y, self.ghost_near_reward)
            
            # Rerun value iteration
            self.value_iteration()
            
            # Update tracking variables
            self.last_food_count = len(food)
            self.last_score = current_score
            self.last_ghost_states = ghost_states
        
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