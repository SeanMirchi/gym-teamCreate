import sys
import gym
from gym import spaces, utils
from gym.utils import seeding
from six import StringIO
from contextlib import closing
import numpy as np

"""
    Observation: 
        Type: Box(3)
        Num	Observation                         Min             Max
        0	Number of selected players          0               3
        1	Current Budget                      -60             115
        2	Current Score                       -15             812
        
        Min -60 because most expensive players that can select is 5 and 8 resulting in -60 budget and end of episode.
        Max 812 for score won't be possible, but just to be sure.
        Min -15 for score because lowest possible players to chose would be 15 + 0 - 30 = -15
        
    Actions: 
        Type: Discrete(10)
        There are 10 discrete actions, selecting from player 1 to 10.
        
    Episode Termination:
        Agent select 3 players
        We are over budget
        Episode length is greater than 200
        
    Rewards:
        By selecting a player we reward agent as much as players score
        If agent selects the same player again we punish with -300
        If we come gone over budget the we punish with -1000
        If agent successfully selects 3 different players, we reward with +500
        
    Players:
        		score	value
        A	0	230	    80
        B	1	289	    50
        C	2	67	    70
        D	3	15	    25
        E	4	68	    90
        F	5	57	    30
        G	6	221	    50
        H	7	0	    85
        I	8	293	    35
        J	9	-30	    50
        
    Facts:
        Best solution is seleting players 8 , 1 and 3  = I B D
        Best Score of 597
"""

class PlayerSelectorEnv(gym.Env):  
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self):
        self.nS = 1000
        self.isd = np.zeros(self.nS)
        self.nA = 10
        self.selectedPlayers = np.array([])
        
        high = np.array([3,115,1000])

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.state = (0,115,0)
        self.lastaction=None
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        playersCount, currentBudget, currentScore = self.state
        reward = 1
        done = False
        
        playerName, playerScore, playerValue = self.mapPlayers(action)
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
            reward = -1000
        elif newPlayerCount == 3:
            done = True
            reward = 500
            
        self.state = (newPlayerCount, newBudget, newScore)
        self.lastaction = action
        return np.array(self.state), reward, done, {}
    
    def mapPlayers(self, playerId):
        if playerId == 0:
            return 'A', 230, 80
        elif playerId == 1:
            return 'B', 289, 50
        elif playerId == 2:
            return 'C', 67, 70
        elif playerId == 3:
            return 'D', 15, 25
        elif playerId == 4:
            return 'E', 68, 90
        elif playerId == 5:
            return 'F', 57, 30
        elif playerId == 6:
            return 'G', 221, 50
        elif playerId == 7:
            return 'H', 0, 85
        elif playerId == 8:
            return 'I', 293, 35
        elif playerId == 9:
            return 'J', -30, 50
        
    def isPlayerAlreadySelected(self, playerName):
        if playerName in self.selectedPlayers:
            return True
        else:
            return False
    
    def reset(self):
        self.seed()
        self.selectedPlayers = np.array([])
        self.isd = np.zeros(self.nS)
        self.state = (0,115,0)
        self.lastaction = None
        return self.state

    def render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        pointer_row, pointer_col, hasPlayer = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        
        out[1 + pointer_row][pointer_col + 1] = utils.colorize(out[1 + pointer_row][pointer_col + 1], 'yellow', highlight=True)

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({}),".format(pointer_row))
            outfile.write("  ({})".format(pointer_col))
            outfile.write("  ({})\n".format(hasPlayer))
            outfile.write("  ({})\n".format(["Down", "Up", "Right", "Left", "Player"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()