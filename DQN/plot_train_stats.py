
import matplotlib.style
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from pathlib import Path
import h5py


matplotlib.style.use('ggplot') # for emulating the aesthetics of ggplot (a popular plotting package for R).

dqn_stats_file = "train-results-2018-06-24-1203/train_stats_breakout.h5"
dqn_double_q_stats_file = "train-results-2018-07-01-2247-double_q/train_stats_breakout.h5"

files = [dqn_stats_file, dqn_double_q_stats_file]
legends =['dqn', 'dqn with double q learning']
plt.figure()
for idx, stat_file in enumerate(files):
  hf = h5py.File(stat_file,'r')
  epoch_stats = hf.get('epoch_stats')
  average_episode_lengths = epoch_stats.get('average_episode_lengths').value
  average_episode_rewards = epoch_stats.get('average_episode_rewards').value
  plt.plot(average_episode_rewards,label=legends[idx])

plt.xlabel('Epoch (each epoch consists of 100,000 steps)')
plt.ylabel('Average reward per episode')
plt.legend
plt.savefig('Figs/training_stats.png')
plt.show()



