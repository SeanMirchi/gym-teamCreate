import sys
import gym
from gym import spaces, utils
from gym.utils import seeding
from six import StringIO
from contextlib import closing
import numpy as np
import pandas as pd

"""
    Version 2 of Player selector, using real players this time.
    But still only defenders for now.
    Instead of selecting 3 players, we are selecting 11.
    This version will read actions from a CSV containing a list of players.
    Will map each action to index of DataFrame.
    All the rules stays the same for now.
    
    Observation: 
        Type: Box(3)
        Num	Observation                         Min             Max
        0	Number of selected players          0               11
        1	Current Budget                      0               1500
        2	Current Score                       0               3000
        
        We don't care about assigning state to minus budget as it is a fail state
        
    Actions: 
        Type: Discrete(73)
        There are 73 discrete actions, selecting from player 0 to 72.
        
    Episode Termination:
        Agent selects 11 players
        We are over budget
        Episode length is greater than 200
        
    Rewards:
        By selecting a player we reward agent as much as players score
        If agent selects the same player again we punish with -300
        If we gone over budget then we punish with -500
        If agent successfully selects 11 different players, we reward with +500
        
    Players data shape:
        		index, name, value, score
        
"""
NUMBER_PLAYERS_TO_SELECT = 11
INITIAL_BUDGET = 1500   
MAX_IMPOSSIBLE_SCORE = 3000

class PlayerSelector2Env(gym.Env):  
    metadata = {'render.modes': ['human', 'ansi']}
    
    def readPlayerData(self):
        my_data = pd.read_csv('playerselector2_players.csv')
        return my_data
    
    def __init__(self):
        self.players = self.readPlayerData()
        self.nA = len(self.players)
        self.selectedPlayers = np.array([])
        
        high = np.array([
                NUMBER_PLAYERS_TO_SELECT,
                INITIAL_BUDGET,
                MAX_IMPOSSIBLE_SCORE
                ])

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.state = (0,INITIAL_BUDGET,0)
        self.lastaction=None
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        playersCount, currentBudget, currentScore = self.state
        reward = 1
        done = False
        
        playerName, playerValue, playerScore  = self.mapPlayers(action)
        playerAlreadySelected = self.isPlayerAlreadySelected(playerName)
        
        if playerAlreadySelected:
            reward = -300
            self.lastaction = action
            return np.array(self.state), reward, done, {}
        else:
            reward = playerScore
            self.selectedPlayers = np.append(self.selectedPlayers, playerName)
        
        newPlayerCount = self.selectedPlayers.size
        newBudget = currentBudget - playerValue
        newScore = currentScore + playerScore
        
        # Check done states
        if newBudget < 0:
            done = True
            reward = -500
        elif newPlayerCount == NUMBER_PLAYERS_TO_SELECT:
            done = True
            reward = 500
            
        self.state = (newPlayerCount, newBudget, newScore)
        self.lastaction = action
        return np.array(self.state), reward, done, {}
    
    def mapPlayers(self, playerId):
        return self.players.iloc[playerId,:]
        
    def isPlayerAlreadySelected(self, playerName):
        if playerName in self.selectedPlayers:
            return True
        else:
            return False
    
    def reset(self):
        self.seed()
        self.selectedPlayers = np.array([])
        self.state = (0,INITIAL_BUDGET,0)
        self.lastaction = None
        return self.state