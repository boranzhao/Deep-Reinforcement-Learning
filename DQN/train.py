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
# from keras.layers import Dense, Dropout,Activation, Flatten
# from keras.layers import Conv2D
# from keras import optimizers

import sys
p = str(Path(__file__).resolve().parents[1])
if p not in sys.path:
    sys.path.append(p)

from collections import defaultdict,namedtuple
from dqn import DQN_Agent, EpisodeStats, EpochStats,Transition
from atari.helper import AtariEnvWrapper

matplotlib.style.use('ggplot') # for emulating the aesthetics of ggplot (a popular plotting package for R).

# Make the environment
env = gym.envs.make('Breakout-v4')

# Wrap the environment so that two boolean variables "done" and "game_over" are output in each step to denote the status of a game, 
# and the reward in each step is clipped into [0,1]
# done: True means that a life is lost
# game_over: True means that the game is over, i.e. all lives have been lost 
env = AtariEnvWrapper(env)     


def train_dqn(env,                  
              num_epoches= 200,
              print_freq = 100,
              epoch_steps = 100_000,  # default value in DQN paper: 100,000
              eval_episodes = 10,                  
              batch_size=32,
              record_video_every_episodes=5000,                  
              save_freq = 100_000,
              double_q = True):
  """
  Train an dqn agent to play the game defined by env

  Args:
      env: OpenAI gym environment     
      num_epoches: Number of epoches to run for, one epoch consists of epoch_steps steps
      print_freq: Print the training info after this many episodes
      epoch_steps: The number of steps in an epoch. The model will be evaluate after this many steps, also 
      eval_episodes: Play this many episodes in evaluting the performance of the dqn model, 
      record_video_every_episodes: Record a video every N episodes
      save_freq: Save the trained model and results every this many steps
      double_q: Use double q learning when this variable is True

  Returns:
      None
  """
  dqn_agent = DQN_Agent(env,double_q=double_q)

  # Initialize the replay memory 
  dqn_agent.initialize_replay_memory()

  # Keep track of the epoch statisticvi s 
  epoch_stats = EpochStats(average_episode_lengths =[], average_episode_rewards=[],
                          train_time=[],evaluation_time=[])
  epoch = 0

  if dqn_agent.double_q:  
    train_results_folder = os.path.join(os.getcwd(),"train-results-"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M")+'-double_q')
  else:
    train_results_folder = os.path.join(os.getcwd(),"train-results-"+datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))

  if not os.path.exists(train_results_folder):
    os.makedirs(train_results_folder)
  trained_model_file_path = os.path.join(train_results_folder,'dqn_model_'+dqn_agent.env.spec._env_name.lower()+'.h5')
  train_stats_file_path =  os.path.join(train_results_folder,'train_stats_'+dqn_agent.env.spec._env_name.lower()+'.h5')
  
  # Creat the directory for monitoring and saving the train results 
  monitor_folder = os.path.join(train_results_folder,"monitor")
  if not os.path.exists(monitor_folder):
    os.makedirs(monitor_folder)
  # Record videso 
  # Add env Monitor wrapper
  loss = float('inf')
  # dqn_agent.env = Monitor(dqn_agent.env,directory=monitor_folder, video_callable=lambda count: count % record_video_every_episodes == 0, resume = True)

  # parameter_updates = 0 # Number of parameter updates so far
  step = 0             # Number of steps so far 
  start_time = time.time()
  episode = 1
  episode_reward = 0
  
  for epoch in range(1,num_epoches+1):
    epoch_start_time = time.time()
    observation = dqn_agent.env.reset()    
    state = dqn_agent.atari_processor.make_initial_state(observation)   
    for _ in range(1, epoch_steps+1):
      step += 1
      #dqn_agent.env.render()
      # Epsilon for this time step
      epsilon = dqn_agent.get_epsilon_for_step(step-1)

      # Take one step
      # Select an action according to epsilon-greedy policy
      action = dqn_agent.epsilon_greedy_policy(np.expand_dims(state,axis=0),epsilon)  # The input to the Conv2D needs to have the shape (sample,rows,cols,channels)
      next_observation,reward,game_over,done = dqn_agent.env.step(action)           
      next_state = dqn_agent.atari_processor.make_next_state(state,next_observation)
      
      # Update episode reward
      episode_reward += reward
      
      # Save transition to replay memory
      dqn_agent.replay_memory.add(Transition(state,action,reward,next_state,done))
     
      # Do a Q-learning update after update_freq steps      
      if step % dqn_agent.update_freq == 0:
        loss = dqn_agent.q_learning_update()     
      
      # Update the target estimator after update_q_target_freq steps
      if step % (dqn_agent.update_q_target_freq) == 1:    
        dqn_agent.copy_model_parameters()
        print("\nCopied model parameters from the estimator to the target network.")

      # Save the train results after save_freq steps
      if step % save_freq == 0:
        dqn_agent.save_train_results(trained_model_file_path,train_stats_file_path,epoch_stats, time.time()-start_time)
      
      # End the episode whenever a life is lost for training
      if done:
        # Print out which step we're on, useful for debugging.
        if episode % print_freq == 0:
          print("Total Step: {}, Episode: {}, epsilon: {:.4f}, reward: {}, loss: {:.4e} \n".format(
                  step, episode, epsilon,episode_reward,loss), end="")
          sys.stdout.flush()
        # Start the next episode without resetting the game
        episode_reward = 0
        episode +=1

      # Reset the game if it is over
      if game_over:
        observation = dqn_agent.env.reset()    
        state = dqn_agent.atari_processor.make_initial_state(observation)   
      else:
        state = next_state 


    # Evaluate the performance of the DQN agent after epoch_steps steps
    epoch_train_time = time.time()-epoch_start_time
    average_episode_reward,average_episode_length = dqn_agent.evaluation(eval_episodes,epsilon_eval=0.05)
    epoch_eval_time= time.time()-epoch_start_time-epoch_train_time

    epoch_stats.average_episode_rewards.append(average_episode_reward)
    epoch_stats.average_episode_lengths.append(average_episode_length)
    epoch_stats.train_time.append(epoch_train_time)
    epoch_stats.evaluation_time.append(epoch_eval_time)
    print("Epoch:{}, Average episode reward:{}, Average episode length:{}, Train Time:{:.2f} mins, Evaluation Time:{:.2f} mins".format(epoch,
          average_episode_reward,average_episode_length,epoch_train_time/60,epoch_eval_time/60))
  
  return epoch_stats

if __name__ ==  '__main__': 
  epoch_stats = train_dqn(env,num_epoches=200,record_video_every_episodes=1000)

  # load the statistics of the episodes
  # f = open("episode_stats.pickle",'rb')
  # pickle.load(f)
  # f.close()

  # print("\nReward of the last episode: {}".format(stats.episode_rewards[-1]))