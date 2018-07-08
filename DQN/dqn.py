import gym
import numpy as np
import random
import itertools
import os
from pathlib import Path

from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout,Activation, Flatten
from keras.layers import Conv2D
from keras import optimizers, losses
import keras.backend as K

import sys
p = str(Path(__file__).resolve().parents[1])
if p not in sys.path:
    sys.path.append(p)


from collections import defaultdict,namedtuple
from skimage import color, transform, exposure
import h5py

from replay_memory import ReplayMemory

Transition = namedtuple("Transition",["state","action","reward","next_state","done"])
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
EpochStats = namedtuple("Stats",["average_episode_lengths", "average_episode_rewards","train_time","evaluation_time"])

class DQN_Agent():
  def __init__(self,
               env,                 
               double_q = True,
               replay_memory_size = 1_000_000, # default value in DQN paper:1,000,000
               replay_memory_init_size = 50_000, # default value in DQN paper: 50,000 
               update_q_target_freq=10_000, # default value in DQN paper: 10,000 
               discount_factor =0.99, 
               epsilon_start=1.0, 
               epsilon_end =0.1, 
               epsilon_decay_step = 1_000_000, # default value in DQN paper:1,000,000
               update_freq = 4,
               batch_size = 32,
               for_train = True):

    """
    A deep-Q-network agent

    Args:
        env: OpenAI gym environment
        double_q: Whether to use double q learning to mitigate the overestimation problem in classical dqn 
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sample when initializing 
            the replay memory.
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
            Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of total_steps to decay epsilon over
        action_repeat_times: Repeat each action selected by the agent this many times. Use a value of 4 results in 
                              the agent seeing only every 4th input frame
        update_freq: The number of actions selected by the agent between successive SGD updates. Using a value of 4
                          results in the agent selecting 4 actions between each pair of successive updates. 
        eval_freq:  Evaluate the model by playing a number of games after this many total_steps
        eval_steps:  Play this many total_steps in evaluting the model, also the number of total_steps in an epoch
        batch_size: Size of batches to sample from the replay memory
        for_train: Denote whether the agent is for training or for playing
    """
    self.env = env
    self.double_q = double_q
    self.replay_memory = ReplayMemory(replay_memory_size)  # Create the replay memory
    self.replay_memory_init_size = replay_memory_init_size    
    self.update_q_target_freq= update_q_target_freq
    self.discount_factor = discount_factor 
    self.epsilon_start= epsilon_start 
    self.epsilon_end = epsilon_end
    self.epsilon_decay_step = epsilon_decay_step
    self.update_freq = update_freq
    self.batch_size = batch_size

    # The epsilon decay schedule
    self._epsilons = np.linspace(epsilon_start,epsilon_end,epsilon_decay_step)
    
    self.nA = self.env.action_space.n
    self.VALID_ACTIONS = list(range(self.nA))

    # Create the atari processor
    self.atari_processor = AtariProcessor()
    
  
    # Create the q_estimator
    self.q_estimator = Q_Estimator(dense2 = self.nA, for_train = for_train)

    if for_train: 
      # Create the q_target 
      self.q_target = Q_Estimator(dense2 = self.nA)

  def get_epsilon_for_step(self,step):
    return self._epsilons[min(step, self.epsilon_decay_step-1)]
  
  def copy_model_parameters(self):
    """
    Copy model parameters of q_estimator to q_target 
    """
    self.q_target.model.set_weights(self.q_estimator.model.get_weights())

  def epsilon_greedy_policy(self,state,epsilon):
    action_probs = np.ones(self.nA, dtype=float)*epsilon/self.nA
    q_values = self.q_estimator.predict_q_values(state)
    best_action = np.argmax(q_values)
    action_probs[best_action] += 1-epsilon
    # Sample an action 
    action = np.random.choice(self.VALID_ACTIONS,p=action_probs)
    return action 
  
  def greedy_policy(self,state):
    q_values = self.q_estimator.predict_q_values(state)
    action = np.argmax(q_values)       
    return action 

  # def predict_q_values(self,state):
  #   return self.q_estimator.predict_q_values(state)

  def q_learning_update(self):
    # Sample a minibatch from the replay_memory 
    samples = self.replay_memory.sample(self.batch_size)
    # print("Len(replay_memory):{}, Len(samples):{}\n".format(len(replay_memory),len(samples)))
    state_batch, action_batch, reward_batch,next_state_batch, done_batch = map(np.array,zip(*samples))
    # zip(*) is used to unzip the list of tuples samples, map is used to convert the resulting tuples (state_batch,...) into numpy array.

    # Calculate q values and targets
    if self.double_q: 
      q_values_next_all_actions = self.q_estimator.predict_q_values(next_state_batch)
      # Get the index of actions corresponding to maximum q values
      max_index = np.argmax(q_values_next_all_actions,axis=1)
      # Get the q value from the target network 
      q_values_next = self.q_target.predict_q_values(next_state_batch)[range(self.batch_size),max_index]  
    else:
      q_values_next_all_actions = self.q_target.predict_q_values(next_state_batch)
      q_values_next = np.amax(q_values_next_all_actions,axis=1)
    
    target_batch = reward_batch + np.invert(done_batch).astype(np.float32)*self.discount_factor*q_values_next
    # note that for q_values_next_all_actions, the first axis (axis=0) is for sample, the second axis (axis=1) is for different actions

    # Perform one gradient descent update using the minibatch sampled from the replay_memory       
    loss = self.q_estimator.minibatch_update(state_batch,action_batch,target_batch)
    return loss   

  def initialize_replay_memory(self):
    # Initialize the replay_memory using randomly selected actions
    observation = self.env.reset()
    state = self.atari_processor.make_initial_state(observation)
    for _ in range(self.replay_memory_init_size):
      # env.render()
      action = random.sample(self.VALID_ACTIONS,1)[0]
      next_observation,reward,game_over,done = self.env.step(action)
      next_state = self.atari_processor.make_next_state(state,next_observation) 
      self.replay_memory.add(Transition(state,action,reward,next_state,done))
      # reset the game if it is over
      if game_over:
        observation = self.env.reset()
        state = self.atari_processor.make_initial_state(observation)
      else:
        state = next_state
  
  def evaluation(self,eval_episodes,epsilon_eval = 0.05):    
    episode_stats  = EpisodeStats(np.zeros(eval_episodes),np.zeros(eval_episodes))
    for i_episode_eval in range(eval_episodes):
      observation = self.env.reset()    
      state = self.atari_processor.make_initial_state(observation) 
      for t_eval in itertools.count():
        #observation = self.env.render()
        # Select an action according to epsilon-greedy policy
        action = self.epsilon_greedy_policy(np.expand_dims(state,axis=0),epsilon_eval)  # The input to the Conv2D needs to have the shape (sample,rows,cols,channels)
        # Take one step
        next_observation,reward,game_over,_ = self.env.step(action)    
        
        # Update the episode statistics
        episode_stats.episode_rewards[i_episode_eval] += reward
        episode_stats.episode_lengths[i_episode_eval] = t_eval

        if game_over:
          break            
        next_state = self.atari_processor.make_next_state(state,next_observation) 
        state = next_state
            
    # Calculate the average score and length 
    average_episode_reward = np.mean(episode_stats.episode_rewards)
    average_episode_length = np.mean(episode_stats.episode_lengths)

    return average_episode_reward, average_episode_length

  def save_train_results(self,trained_model_file_path,train_stats_file_path,epoch_stats,time_elapse):
    # Save the trained model
    self.q_estimator.model.save(trained_model_file_path)

    # Save epoch stats   
    hf = h5py.File(train_stats_file_path,'w')
    g1 = hf.create_group('epoch_stats')
    g1.create_dataset('average_episode_rewards', data = epoch_stats.average_episode_rewards)
    g1.create_dataset('average_episode_lengths', data = epoch_stats.average_episode_lengths)
    g1.create_dataset('train_time', data = epoch_stats.train_time)
    g1.create_dataset('evaluation_time', data = epoch_stats.evaluation_time)
    
    g2 = hf.create_group('other_stats')
    g2.create_dataset('time_elapse', data = time_elapse) 

    hf.close()

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

  def make_next_state(self,state, next_image_color):
    next_image = self.process_image(next_image_color)
    next_state = np.append(state[:,:,1:],next_image,axis=2)  # state[:,:,0] corresponds to the oldest frame
    return next_state


  def make_initial_state(self,image_color):
    """
      Reset the state to be the stacks of four repetitions of the processed image_color
    """
    image = self.process_image(image_color)
    state = np.squeeze(np.stack([image]*4,axis=2),axis=3)
    return state

 
class Q_Estimator():
  """
  A convolution neural network model for estimating the Q values
  """
  def __init__(self,frames=4,conv1 =(32,(8,8),(4,4)), conv2=(64,(4,4),(2,2)), conv3=(64,(3,3),(1,1)), dense1=512, dense2=4, for_train=False):
    # for the convolution layer, e.g. conv1 = (32,(8,8),(4,4)), 32 is the number of filters, (8,8) is the size of a filter, (4,4) is the strides
    # for fully connected (dense) layers, the parameters are the number of outputs
    self.frames=frames
    self.conv1 = conv1
    self.conv2 = conv2
    self.conv3 = conv3
    self.dense1 = dense1
    self.dense2 = dense2

    # q_estimator will be loaded in play mode
    if for_train:
      self.__build_model()
  
  def __build_model(self):    
    self.model = Sequential()
    self.model.add(Conv2D(self.conv1[0],self.conv1[1],strides=self.conv1[2],input_shape=(84,84,4),activation='relu'))
    self.model.add(Conv2D(self.conv2[0],self.conv2[1],strides=self.conv2[2],activation='relu'))
    self.model.add(Conv2D(self.conv3[0],self.conv3[1],strides=self.conv3[2],activation='relu'))

    self.model.add(Flatten())
    self.model.add(Dense(self.dense1,activation='relu'))
    self.model.add(Dense(self.dense2)) # default activation is linear, i.e. a(x) = x


    # optimizer = optimizers.RMSprop(lr=0.00025,rho=0.95,epsilon=0.01)  # lr is the learning rate, value in DQN paper: lr=0.00025
    optimizer = optimizers.Adam(lr=1e-4)  # lr is the learning rate
    self.model.compile(loss=sum_logcosh,optimizer = optimizer)
    print("Built a neural network model!")

  # given a state s, predict the value of Q(s,a) for all actions a
  def predict_q_values(self,state):
    return self.model.predict(state/255)   # /255 is used to scale the elements into [0,1]

  # update the weights in the model using a minibatch
  def minibatch_update(self,state_batch,action_batch,target_batch):

    state_batch = state_batch/255  # /255 is used to scale the elements into [0,1]
    # For each state s in state_batch, get Q(s,a) for all actions a 
    target_batch_all_actions= self.model.predict(state_batch)   # numpy array of shape (batch_size,len(valid_actions))
    # Replacing the target value for Q(s,a), where a is from action_batch (actually executed),
    #  with the value in target_batch, i.e. (reward + discount_factor*max(Q(s',a))
    for i_sample in range(action_batch.shape[0]):
      target_batch_all_actions[i_sample,action_batch[i_sample]] = target_batch[i_sample]     
    return self.model.train_on_batch(state_batch,target_batch_all_actions)  



def sum_logcosh(y_true,y_pred):
  """
  logcosh is similar to huber loss function
  """
  return K.sum(losses.logcosh(y_true,y_pred))