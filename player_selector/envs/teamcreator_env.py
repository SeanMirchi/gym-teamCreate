import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from six import StringIO
from contextlib import closing
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "|=====|",
    "|--   |",
    "|---  |",
    "|-----|",
    "|-    |",
    "|=====|",
    ]
"""

    Observations: 
    There are 20 discrete states since there are 20 actual positions. Only 11 of those positions are valid.
    
    Actions: 
        There are 4 discrete deterministic actions:
        - 0: move down
        - 1: move up
        - 2: move right 
        - 3: move left 
        - 4: select player
"""
def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class TeamCreatorEnv(gym.Env):  
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self):
        num_rows = 4
        num_columns = 5
        
        self.desc = np.asarray(MAP, dtype='c')
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        
        self.nS = 100
        self.isd = np.zeros(self.nS)
        self.lastaction = None # for rendering
        self.nA = 5
        
        self.nPlayers = 0

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        pass
    
    def step(self, action):
        row, col, player = self.decode(self.s)
        goneOutside = False
        reward = -1
        done = False
        hasPlayer = player
        
        if action == 0 and row < self.max_row:
            row = row + 1
        elif action == 0 and row == self.max_row:
            row = self.max_row
            reward = -10
            goneOutside = True
        elif action == 1 and row > 0:
            row = row -1
        elif action == 1 and row == 0:
            row = 0
            reward = -10
            goneOutside = True
        elif action == 2 and col < self.max_col:
            col = col + 1
        elif action == 2 and col == self.max_col:
            col = self.max_col
            reward = -10
            goneOutside = True
        elif action == 3 and col > 0:
            col = col - 1
        elif action == 3 and col == 0:
            col = 0
            reward = -10
            goneOutside = True
        elif action == 4 and self.desc[row + 1, col + 1] == b"-":
            self.desc[row + 1, col + 1] = "P"
            self.nPlayers+=1
            reward = 2 * self.nPlayers
            hasPlayer = 1
#        elif action == 4 and self.desc[row + 1, col + 1] != b"-":
#            reward = -10
            
#        if action != 4 and goneOutside == False and self.desc[row + 1, col + 1] == b"-":
#            reward = 0
#        if action != 4 and goneOutside == False and self.desc[row + 1, col + 1] == b"P":
#            reward = 0
        
        if action != 4 and self.desc[row + 1, col + 1] == b"P":
            hasPlayer = 1
        elif action != 4 and self.desc[row + 1, col + 1] != b"P":
            hasPlayer = 0
            
        # Add done state, when we have 11 players we can call it done for now, and reward a big prize of 100
        if self.nPlayers == 11:
            done = True
            reward = 1000
            
        new_state = self.encode(row, col, hasPlayer)
        self.s = new_state
        self.lastaction = action
        return (self.s, reward, done, {})

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.seed()
        self.isd = np.zeros(self.nS)
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.desc = np.asarray(MAP, dtype='c')
        self.nPlayers = 0
        return self.s
    
    def encode(self, pointer_row, pointer_col, hasPlayer):
        # (5) 5
        i = pointer_row
        i *= 5
        i += pointer_col
        i *= 4
        i += hasPlayer
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

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