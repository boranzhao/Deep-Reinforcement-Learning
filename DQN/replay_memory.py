import random 
import numpy as np

class ReplayMemory():
    def __init__(self,size):
        """Create Replay memory.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the memory. When the memory
            overflows the old memories are dropped.
        """

        self._storage =[]
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self,transition):
        if self._next_idx >= len(self._storage):
            self._storage.append(transition)
        else:
            self._storage[self._next_idx] = transition
        
        self._next_idx = (self._next_idx+1) % self._maxsize   # Do not use pop
    
    def sample(self,batch_size):
        """
        Sample a batch of experiences

        Parameters
        ----------
        batch_size: int 
            How many transitions to sample
        
        Returns
        -------
        A batch of transitions consisting of state, action, reward, next_state, done
        
        """
        return random.sample(self._storage,batch_size)

class PrioritizedReplayMemory(ReplayMemory):
    """
    Create Prioritized Replay Buffer

    Parameters 
    ----------
    size:int 
        Max number of transitions to store in the memory. When the memory overflows 
        the old memories are dropped.
    alpha: float
        how much prioritization is used
        ( 0 - no prioritization, 1 - full prioritization)
    """

    def __init__(self,size,alpha):
        super().__init__(size)
        self._alpha = alpha

    def add(self,transition):
        raise NotImplementedError
    
    def sample(self,batch_size):
        raise NotImplementedError

