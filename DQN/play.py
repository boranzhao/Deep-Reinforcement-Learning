import gym
from gym.wrappers import Monitor
import matplotlib.style
import matplotlib.pyplot as plt
import itertools
import numpy as np
import random
import os
from pathlib import Path
import time
import h5py
import datetime

from keras.models import load_model

import sys
p = str(Path(__file__).resolve().parents[1])
if p not in sys.path:
    sys.path.append(p)

from collections import defaultdict,namedtuple
from dqn import DQN_Agent, EpisodeStats, EpochStats,Transition, sum_logcosh
from atari.helper import AtariEnvWrapper

matplotlib.style.use('ggplot') # for emulating the aesthetics of ggplot (a popular plotting package for R).

# Make the environment
env = gym.envs.make('Breakout-v4')
# env = gym.envs.make('BreakoutDeterministic-v4')

# # Wrap the environment so that two boolean variables "done" and "game_over" are output in each step to denote the status of a game, 
# # and the reward in each step is clipped into [0,1]
# # done: True means that a life is lost
# # game_over: True means that the game is over, i.e. all lives have been lost 
env = AtariEnvWrapper(env)     

def play_dqn(env,
             num_episodes,                  
             trained_model_file_path,
             record_video_every_episodes=1,
             only_load_weights = True):
  """
  Play the game with a pre-trained DQN agent

  Args:
      env: OpenAI environment
      trained_model_file_path: Path of the file storing the trained model
      num_episodes: Number of episodes to run for
      record_video_every_episodes: Record a video every N episodes
      double_q: Use double q learning when this variable is True

  Returns:
      An plotting.EpisodeStats object with two numpy arrays for episode lengths and episode_rewards
  """ 

  print("Build a DQN agent...")
  dqn_agent = DQN_Agent(env, for_train=False)

  # Load the saved model
  print("Load the trained Q-network...") 
  dqn_agent.q_estimator.model = load_model(trained_model_file_path,custom_objects={'sum_logcosh':sum_logcosh})
  

  # Keeps track of episode statistics
  stats = EpisodeStats(
      episode_lengths=np.zeros(num_episodes),
      episode_rewards=np.zeros(num_episodes))
  
  # Creat the directory for monitoring
  # Record videos
  monitor_folder = os.path.join(os.getcwd(),"monitor-play-"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))
  if not os.path.exists(monitor_folder):
    os.makedirs(monitor_folder) 
  # Add env Monitor wrapper
  env = Monitor(env, directory= monitor_folder, video_callable=lambda count: count % (record_video_every_episodes) == 0, resume = True)

  steps = 0
  no_ops = 0    # Number of "no fire" actions at the start of an episode (fire is used to start the game)
  
  for i_episode in range(num_episodes):
    observation = env.reset()

    #### Should remove this later because the agent should learn how to start the game #############
    observation,reward,game_over,_ = env.step(1)

    state = dqn_agent.atari_processor.make_initial_state(observation)
    for t in itertools.count(): 
      env.render()

      # Take one step
      # Select the best action
      action = dqn_agent.greedy_policy(np.expand_dims(state,axis=0))  

      # action = dqn_agent.epsilon_greedy_policy(np.expand_dims(state,axis=0), epsilon=0.05)  
      next_observation,reward,game_over,done = env.step(action)

      # In play mode, an episode terminates when all lives are lost
      if game_over:
        break

      if done:
        next_observation,reward,game_over,done = env.step(1)

      next_state = dqn_agent.atari_processor.make_next_state(state,next_observation)

      # Update statistics
      stats.episode_rewards[i_episode] +=reward
      stats.episode_lengths[i_episode] = steps      
      
      state = next_state
    # Print out which step we're on, useful for debugging.
    print("Total Steps: {}, Steps: {} @ Episode {}/{}, score: {}".format(
            steps, t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode]), end="")
    sys.stdout.flush()
  return stats

if __name__ ==  '__main__': 
  stats = play_dqn(env,
                   num_episodes = 3,                  
                  #  trained_model_file_path = "train-results-2018-07-01-2247-double_q/dqn_model_breakout.h5")
                   trained_model_file_path = "train-results-2018-06-24-1203/dqn_model_breakout.h5")
