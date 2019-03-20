import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

"""
    Version 3 of Player selector, using all players this time.
    We are selecting 11 players, but for a specific formation: 4-3-3.
    So it means 1 GK, 4 DF, 3 MF and 3 ST.
    This version will read actions from a CSV containing a list of players.
    Will map each action to index of DataFrame.
    This will also have a render function to print out selected players by their name.
    All the rules stays the same for now.
    
    Observation: 
        Type: Box(3)
        Num	Observation                         Min             Max
        0	Number of selected players          0               11
        0	Number of selected GK               0               1
        0	Number of selected DF               0               4
        0	Number of selected MF               0               3
        0	Number of selected ST               0               3
        1	Current Budget                     -1500            1500
        2	Current Score                      -10000           10000
        
    Actions: 
        Type: Discrete(417)
        There are 417 discrete actions, selecting from player 0 to 416.
        
    Episode Termination:
        Agent selects 11 players
        We are over budget
        Episode length is greater than 200
        
    Rewards:
        By selecting a player we reward agent as much as players score
        If agent selects the same player again we punish with -300 and ignore this action
        If agent selects a position that is full we punish with -300 and ignore this action
        If we gone over budget then we punish with -1000
        If agent successfully selects 11 different players, we reward with +500
        
    Players data shape:
        		index, name, position, value, score
                
    Note:
        Maximum possible score here is probably 2900
        
"""
MAX_PLAYERS = 11
MAX_GK = 1
MAX_DF = 4
MAX_MF = 3
MAX_ST = 3
INITIAL_BUDGET = 1500   
MAX_IMPOSSIBLE_SCORE = 10000

POSITION_GK = "goalkeeper"
POSITION_DF = "defender"
POSITION_MF = "midfielder"
POSITION_ST = "attacker"

class PlayerSelector3Env(gym.Env):  
    
    def readPlayerData(self):
        my_data = pd.read_csv('playerselector3_players.csv', sep = ';')
        return my_data
    
    def __init__(self):
        self.players = self.readPlayerData()
        self.nA = len(self.players)
        
        high = np.array([
                MAX_PLAYERS,
                MAX_GK,
                MAX_DF,
                MAX_MF,
                MAX_ST,
                INITIAL_BUDGET,
                MAX_IMPOSSIBLE_SCORE
                ])

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.selectedPlayers = np.array([])
        self.state = (0, 0, 0, 0, 0,INITIAL_BUDGET,0)
        self.lastaction = None
        pass

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        playersCount, countGK, countDF, countMF, countST, currentBudget, currentScore = self.state
        reward = 1
        done = False
        
        playerName, playerPosition, playerValue, playerScore  = self.mapPlayers(action)
        playerAlreadySelected = self.isPlayerAlreadySelected(playerName)
        
        self.lastaction = action
        
        if playerAlreadySelected or self.isPositionOverflow(playerPosition):
            reward = 0
            return self.state, reward, done, self.selectedPlayers
        else:
            reward = playerScore
            self.selectedPlayers = np.append(self.selectedPlayers, playerName)
        
        newPlayerCount = self.selectedPlayers.size
        newCountGK = countGK+1 if (playerPosition == POSITION_GK) else countGK
        newCountDF = countDF+1 if (playerPosition == POSITION_DF) else countDF
        newCountMF = countMF+1 if (playerPosition == POSITION_MF) else countMF
        newCountST = countST+1 if (playerPosition == POSITION_ST) else countST
        newBudget = currentBudget - playerValue
        newScore = currentScore + playerScore
        
        # Check done states
        if newBudget < 0:
            done = True
            reward = -1000
        elif newPlayerCount == MAX_PLAYERS:
            done = True
            reward = 500
            
        self.state = (newPlayerCount, newCountGK, newCountDF, newCountMF, newCountST, newBudget, newScore)
        return self.state, reward, done, self.selectedPlayers
    
    def mapPlayers(self, playerId):
        return self.players.iloc[playerId,:]
        
    def isPlayerAlreadySelected(self, playerName):
        if playerName in self.selectedPlayers:
            return True
        else:
            return False
        
    def isPositionOverflow(self, playerPosition):
        _, countGK, countDF, countMF, countST, _, _ = self.state
        if playerPosition == POSITION_GK:
            return countGK == MAX_GK
        elif playerPosition == POSITION_DF:
            return countDF == MAX_DF
        elif playerPosition == POSITION_MF:
            return countMF == MAX_MF
        elif playerPosition == POSITION_ST:
            return countST == MAX_ST
    
    def reset(self):
        self.seed()
        self.selectedPlayers = np.array([])
        self.state = (0, 0, 0, 0, 0,INITIAL_BUDGET,0)
        self.lastaction = None
        return self.state