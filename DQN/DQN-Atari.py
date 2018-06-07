task= "train"
import gym
from gym.wrappers import Monitor
import matplotlib
import itertools
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
import pickle

from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout,Activation, Flatten
from keras.layers import Conv2D
from keras import optimizers

import sys
p = str(Path(__file__).resolve().parents[1])
if p not in sys.path:
    sys.path.append(p)


from collections import defaultdict,namedtuple
# import lib.plotting as plotting
from lib import plotting
from lib.atari.helper import AtariEnvWrapper
# from lib import plotting
from skimage import color, transform, exposure
matplotlib.style.use('ggplot') # for emulating the aesthetics of ggplot (a popular plotting package for R).

#%% environment
env = gym.envs.make('Breakout-v4')

# wrap the environment so that an episode is done when a life is lost, but the game is still reset after losing all lives
env = AtariEnvWrapper(env) 

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 right 
VALID_ACTIONS = [0,1,2,3]

class AtariProcessor():
  def __init__(self,frame_size=(84,84)):
    # self.frames = frames
    self.frame_size = frame_size

  def process_image(self,image_color):
    """
      Process a raw Atari image. Resize it and convert it into grayscale
    """
    # image = color.rgb2gray(image_color)  # (rows,cols), the color channels are removed
    image = color.rgb2yuv(image_color)[:,:,0]  # rgb2y, ignoring the u, v components
    image = transform.resize(image,self.frame_size)
    image = exposure.rescale_intensity(image,out_range=(0,255))
    image = image.astype(np.uint8)           # convert to int 8 so that the replay_memory will not take too much space
    image = image.reshape(image.shape[0],image.shape[1],1)  # the shape (samples,rows,cols, channels) needed for Conv2D, the sample axis will be added later
    return image
  # def process_state(self,image_color):
  #   image = self.process_image(image_color)
  #   return image.reshape(1,image.shape[0],image.shape[1],1)  # the shape (samples,rows,cols, channels) needed for Conv2D

  # def reset_state(self,image_color):
  #   """
  #     Reset the state to be the stacks of four repetitions of the processed image_color
  #   """
  #   image = self.process_image(image_color)
  #   self.state= np.stack([image]*self.frames, axis=2)


  # def update_state(self,image_color):
  #   image = self.process_image(image_color)
  #   self.state = np.append(self.state[:,:,1:],image,axis=2)
  
def make_epislon_greedy_policy(estimator,nA):
  def policy_fn(state,epsilon):
    action_probs = np.ones(nA, dtype=float)*epsilon/nA
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
    action_probs[best_action] += 1-epsilon
    return action_probs
  return policy_fn  
  
class Q_Estimator():
  """
  A convolution neural network for Q-value estimation
  """
  def __init__(self,frames=4,conv1 =(32,(8,8),(4,4)), conv2=(64,(4,4),(2,2)), conv3=(64,(3,3),(1,1)), dense1=512, dense2=len(VALID_ACTIONS)):
    # for the convolution layer, e.g. conv1 = (32,(8,8),(4,4)), 32 is the number of filters, (8,8) is the size of a filter, (4,4) is the strides
    # for fully connected (dense) layers, the parameters are the number of outputs
    self.frames=frames
    self.conv1 = conv1
    self.conv2 = conv2
    self.conv3 = conv3
    self.dense1 = dense1
    self.dense2 = dense2

    self.__build_model()
  
  def __build_model(self):
    print("Now we build the convoulution neural network model")
    self.model = Sequential()
    self.model.add(Conv2D(self.conv1[0],self.conv1[1],strides=self.conv1[2],input_shape=(84,84,4),activation='relu'))
    self.model.add(Conv2D(self.conv2[0],self.conv2[1],strides=self.conv2[2],activation='relu'))
    self.model.add(Conv2D(self.conv3[0],self.conv3[1],strides=self.conv3[2],activation='relu'))

    self.model.add(Flatten())
    self.model.add(Dense(self.dense1,activation='relu'))
    # self.model.add(Dropout(rate=0.2))  # using Dropout to prevent overfitting
    self.model.add(Dense(self.dense2)) # default activation is linear, i.e. a(x) = x
    # self.model.add(Dropout(rate=0.2))

    optimizer = optimizers.RMSprop(lr=0.00025,rho=0.95,epsilon=0.01)  # lr is the learning rate, value in DQN paper: lr=0.00025
    # optimizer = optimizers.Adam(lr=1e-4)  # lr is the learning rate
    self.model.compile(loss='mean_squared_error',optimizer = optimizer)
    print("We finish building the model.")

  # given a state s, predict the value of Q(s,a) for all actions a
  def predict(self,state):
    return self.model.predict(state)


  # update the weights in the model using a minibatch
  def minibatch_update(self,state_batch,action_batch,target_batch):
    # For each state s in state_batch, get Q(s,a) for all actions a 
    target_batch_all_actions= self.model.predict(state_batch)   # numpy array of shape (batch_size,len(valid_actions)) 
    # Replacing the target value for Q(s,a), where a is from action_batch (actually executed),
    #  with the value in target_batch equal to (reward + discount_factor*max(Q(s',a))
    for i_sample in range(action_batch.shape[0]):
      target_batch_all_actions[i_sample,action_batch[i_sample]] = target_batch[i_sample]
    return self.model.train_on_batch(state_batch,target_batch_all_actions)


def copy_model_parameters(estimator1,estimator2):
  """
  Copy model parameters of one estimator to another 

  Args:
    estimator1: Estimator to copy the parameters from
    estimator2: Estimator to cpy the parameters to
  """
  estimator2.model.set_weights(estimator1.model.get_weights())

def deep_q_learing(env,
                  atari_processor,
                  q_estimator,
                  num_games=100000,
                  replay_memory_size = 1000000, # default value in DQN paper 
                  replay_memory_init_size = 50000, # default value in DQN paper  
                  update_target_estimator_every=10000, # default value in DQN paper 
                  discount_factor =0.99, 
                  epsilon_start=1.0, 
                  epsilon_end =0.1, 
                  epsilon_decay_step = 1000000, # default value in DQN paper 
                  action_repeat_times = 1,      # default value in DQN paper is 4; however, in their Lua codes, I did not find this parameter. Therefore, I set this to 1.
                  update_frequency = 4,
                  batch_size=32,
                  record_video_every_episodes=500,
                  
                  save_model_every = 100000):
  """
  Q-Learning algorithm for off-policy TD control using a neural network to approximate the Q-value function

  Args:
      env: OpenAI environment
      atari_processor: A AtariProcessor object
      q_estimator: Estimator object used for the Q values
      num_games: Number of episodes to run for
      replay_memory_size: Size of the replay memory
      replay_memory_init_size: Number of random experiences to sample when initializing 
          the reply memory.
      discount_factor: Gamma discount factor
      epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
      epsilon_end: The final minimum value of epsilon after decaying is done
      epsilon_decay_steps: Number of steps to decay epsilon over
      action_repeat_times: Repeat each action selected by the agent this many times. Use a value of 4 results in 
                            the agent seeing only every 4th input frame
      update_frequency: The number of actions selected by the agent between successive SGD updates. Using a value of 4
                        results in the agent selecting 4 actions between each pair of successive updates. 
      batch_size: Size of batches to sample from the replay memory
      record_video_every_episodes: Record a video every N episodes
      save_model_every: Save the weights of the model every N steps

  Returns:
      An plotting.EpisodeStats object with two numpy arrays for episode lengths and episode_rewards
  """
  Transition = namedtuple("Transition",["state","action","reward","next_state","done"])
  # Initialize the replay memory for storing the transitions [state,action,reward,new_state]:
  replay_memory =[]  


  # Create another q_estimator to be used for computing the q-target. 
  target_estimator = Q_Estimator()
  # target_estimator = clone_model(q_estimator.model)
  copy_model_parameters(q_estimator,target_estimator)

  # Keeps track of useful statistics
  stats = plotting.EpisodeStats(
      episode_lengths=np.zeros(num_games),
      episode_rewards=np.zeros(num_games))


  # Creat the directory for monitoring
  monitor_path = os.path.join(os.getcwd(),"monitor")
  if not os.path.exists(monitor_path):
    os.makedirs(monitor_path)

  # The epsilon decay schedule
  epsilons = np.linspace(epsilon_start,epsilon_end,epsilon_decay_step)


  # Initialize the replay_memory using randomly selected actions
  observation = env.reset()
  state = atari_processor.process_image(observation)
  state = np.squeeze(np.stack([state]*4,axis=2),axis=3)
  for _ in range(replay_memory_init_size):
    # env.render()
    action = random.sample(VALID_ACTIONS,1)[0]
    next_observation,reward,game_over,done = env.step(action)
    next_observation = atari_processor.process_image(next_observation)
    next_state = np.append(state[:,:,1:],next_observation,axis=2)   # state[:,:,0] corresponds to the oldest frame
    replay_memory.append(Transition(state,action,reward,next_state,done))
    # if done:
    #   print("Lost a life\n")
    if game_over:
      observation = env.reset()
      state = atari_processor.process_image(observation)
      state = np.squeeze(np.stack([state]*4,axis=2),axis=3)
    else:
      state = next_state

  # Record videso 
  # Add env Monitor wrapper
  loss = float('inf')
  env = Monitor(env,directory=monitor_path, video_callable=lambda count: count % record_video_every_episodes == 0, resume = True)
  

  action_selections = 0 # Number of action selections so far
  parameter_updates = 0 # Number of parameter updates so far
  steps = 0             # Number of steps so far 
  for i_game in range(num_games):
    observation = env.reset()
    state = atari_processor.process_image(observation)
    state = np.squeeze(np.stack([state]*4,axis=2),axis=3)    
    for _ in itertools.count():
      observation = env.render()
      ############################# Play ################################
      # Epsilon for this time step
      epsilon = epsilons[min(action_selections, epsilon_decay_step-1)]

      # Update the target estimator after update_target_estimator_every steps
      if parameter_updates % (update_target_estimator_every) == 0:    
        copy_model_parameters(q_estimator,target_estimator)
        print("\nCopied model parameters to target network.")

      if parameter_updates % save_model_every == 0:
        q_estimator.model.save_weights('dqn_breakout_weights.h5')

      # Get the policy corresponding to the current q_estimator
      policy = make_epislon_greedy_policy(q_estimator,len(VALID_ACTIONS))
            
      # Take one step
      # Select an action according to epsilon-greedy policy
      action_probs = policy(np.expand_dims(state,axis=0),epsilon)  # The input to the Conv2D needs to have the shape (sample,rows,cols,channels)
      action = np.random.choice(VALID_ACTIONS,p=action_probs)
      action_selections +=1

      # Execute the action selected #action_repeat_times# times
      for _ in range(action_repeat_times):
        if done:
          print("Lost a life")
        next_observation,reward,game_over,done = env.step(action)
        steps += 1
        if game_over:
          break
      next_observation = atari_processor.process_image(next_observation)
      next_state = np.append(state[:,:,1:],next_observation,axis=2)

      # Pop the first element if the replay_memory is 
      if len(replay_memory) == replay_memory_size:
        replay_memory.pop(0)
      # Save transition to replay memory
      replay_memory.append(Transition(state,action,reward,next_state,game_over))

      # Update statistics
      stats.episode_rewards[i_game] +=reward
      stats.episode_lengths[i_game] = steps

      # Print out which step we're on, useful for debugging.
      print("Step: {}, Action-Sels: {} @ Game {}/{}, epsilon: {:.4f}, reward: {}, loss: {:.4e} \n".format(
              steps,action_selections, i_game + 1, num_games, epsilon,stats.episode_rewards[i_game],loss), end="")
      sys.stdout.flush()
       
      if action_selections % 4 == 0:
        ########################### Learn ###########################
        # Sample a minibatch from the replay_memory 
        samples = random.sample(replay_memory,batch_size)
        # print("Len(replay_memory):{}, Len(samples):{}\n".format(len(replay_memory),len(samples)))
        state_batch, action_batch, reward_batch,next_state_batch, done_batch = map(np.array,zip(*samples))
          # zip(*) is used to unzip the list of tuples samples, map is used to convert the resulting tuples (state_batch,...) into numpy array.

        # Calculate q values and targets
        q_values_next = target_estimator.predict(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) \
                        *discount_factor*np.amax(q_values_next,axis=1)
        # note that for q_values_next, the first axis (axis=0) is for sample, the second axis (axis=1) is for different actions

        # Perform one gradient descent update using the minibatch sampled from the replay_memory      
        loss = q_estimator.minibatch_update(state_batch,action_batch,target_batch)
        parameter_updates += 1
      
      if game_over:
        break
      state = next_state
  # Save the statistics of the episodes
  # f = open("episode_stats.pickle","wb")
  # pickle.dump(stats,f)
  # f.close()
  return stats


def play_with_trained_model(env,
                  atari_processor,
                  q_estimator,
                  num_games,                  
                  model_file_path,
                  action_repeat_times = 4,
                  record_video_every_episodes=50):
  """
  Play the game with a pre-trained DQN model

  Args:
      env: OpenAI environment
      atari_processor: A AtariProcessor object
      q_estimator: Estimator object used for the Q values
      model_file_path: Path of the file storing the model weights
      num_games: Number of episodes to run for
      action_repeat_times: Repeat each action selected by the agent this many times. Use a value of 4 results in 
                            the agent seeing only every 4th input frame
      record_video_every_episodes: Record a video every N episodes

  Returns:
      An plotting.EpisodeStats object with two numpy arrays for episode lengths and episode_rewards
  """

  print("Play with a trained DQN...\n")

  # Load the saved model weights
  q_estimator.model.load_weights(model_file_path)

  # Keeps track of useful statistics
  stats = plotting.EpisodeStats(
      episode_lengths=np.zeros(num_games),
      episode_rewards=np.zeros(num_games))
  
  # Creat the directory for monitoring
  # Record videos
  monitor_path = os.path.join(os.getcwd(),"monitor-play")
  if not os.path.exists(monitor_path):
    os.makedirs(monitor_path) 
  # Add env Monitor wrapper
  env = Monitor(env,directory= monitor_path, video_callable=lambda count: count % record_video_every_episodes == 0, resume = True)

  action_selections = 0
  steps = 0
  no_ops = 0    # Number of "no fire" actions at the start of an episode (fire is used to start the game)
  for i_game in range(num_games):
    observation = env.reset()
    state = atari_processor.process_image(observation)
    state = np.squeeze(np.stack([state]*4,axis=2),axis=3)
    for _ in itertools.count(): 
      # if action_selections % 10 == 0:
      #   _,_,game_over,done= env.step(1) # FIRE to start the game to avoid longlasting pause
      #   if game_over:
      #     break     
      observation = env.render()  

      # Take one step
      # Select the best action
      q_values = q_estimator.predict(np.expand_dims(state,axis=0))
      action = np.argmax(q_values)  
      action_selections +=1

      # Execute the action selected #action_repeat_times# times
      for _ in range(action_repeat_times):
        next_observation,reward,game_over,done = env.step(action)
        steps +=1
        if game_over:
          break
      next_observation = atari_processor.process_image(next_observation)
      next_state = np.append(state[:,:,1:],next_observation,axis=2)

      # # Execute the action selected
      # next_observation,reward,game_over,done = env.step(action)
      # next_observation = atari_processor.process_image(next_observation)
      # next_state = np.append(state[:,:,1:],next_observation,axis=2)

      # Update statistics
      stats.episode_rewards[i_game] +=reward
      stats.episode_lengths[i_game] = steps

      # Print out which step we're on, useful for debugging.
      print("Step: {}, Action-Sels: {} @ Episode {}/{}, reward: {}, lives:{}\n".format(
              steps, action_selections, i_game + 1, num_games, stats.episode_rewards[i_game],env.env.env.env.ale.lives()), end="")
      sys.stdout.flush()
      # for debugging
      if done:
        print('Lost a life\n')

      if game_over:
        break
      state = next_state
  return stats


################## Run ###################
# create the atari_processor and q_esitimator
atari_processor = AtariProcessor()
q_estimator = Q_Estimator()

if task == "train": 
  stats = deep_q_learing(env,
                        atari_processor,
                        q_estimator,
                        num_games=10000,
                        replay_memory_size = 500000, # To avoid out-of-memory problem, DQN paper: 1 million
                        replay_memory_init_size = 5000, # DQN paper: 50000
                        update_target_estimator_every=5000, # DQN paper: 10000
                        discount_factor =0.99, 
                        epsilon_start=1.0, 
                        epsilon_end =0.1, 
                        epsilon_decay_step = 100000,  # DQN paper: 1 million
                        action_repeat_times = 4,
                        update_frequency = 4,
                        batch_size=32,
                        record_video_every_episodes=100)

  # load the statistics of the episodes
  # f = open("episode_stats.pickle",'rb')
  # pickle.load(f)
  # f.close()

  # print("\nReward of the last episode: {}".format(stats.episode_rewards[-1]))
elif task =="play":
  stats = play_with_trained_model(env,
                                  atari_processor,
                                  q_estimator,
                                  num_games=2,                                  
                                  model_file_path="dqn_breakout_weights_back.h5",
                                  action_repeat_times = 1,
                                  record_video_every_episodes=50)


# from gym.utils.play import play
# play(env)

# env.reset()
# for _ in range(1000):
#   observation =env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, _= env.step(action)