# original code is from https://github.com/dennybritz/reinforcement-learning/blob/master/lib/atari/helpers.py
# slightly modified so that both "done" means the game is over, loss_life means a life is lost 

import numpy as np


class AtariEnvWrapper(object):
  """
  Wraps an Atari environment to end an episode when a life is lost.
  """
  def __init__(self, env):
    self.env = env

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, *args, **kwargs):
    
    lives_before = self.env.env.ale.lives()
    next_state, reward, done, info = self.env.step(*args, **kwargs)
    lives_after = self.env.env.ale.lives()

    # # for debugging: the following two values are always the same
    # print("\nenv.ale.lives:{},  info.ale.lives:{}\n".format(lives_after,info['ale.lives']))

    # the "done" from gym actually means game is over
    game_over = done
    # End the episode when a life is lost
    if lives_before > lives_after:
      done = True

    # Clip rewards to [,1]
    # reward = max(min(reward, 1), 0)

    return next_state, reward, game_over, done #



class WrapperToAtari(object):
  """
  Wraps a non-Atari environment so that it works like an Atari environment.
  """
  def __init__(self, env):
    self.env = env

  def __getattr__(self, name):
    return getattr(self.env, name)

  def step(self, *args, **kwargs):    
    
    next_state, reward, done, info = self.env.step(*args, **kwargs)
   
    # # for debugging: the following two values are always the same
    # print("\nenv.ale.lives:{},  info.ale.lives:{}\n".format(lives_after,info['ale.lives']))

    # the "done" from gym actually means game is over
    game_over = done
    
    # Clip rewards to [-1,1]
    reward = max(min(reward, 1), -1)

    return next_state, reward, game_over, done 
