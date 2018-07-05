
import numpy as np
import random
from collections import deque


class ReplayBuffer(object):
    
    def __init__(self, bufferSize, random_seed = 1234):
        
        self.bufferSize = bufferSize
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.bufferSize:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def size(self):
        return self.count
    
    def sampleBatch(self, batchSize):
        
        batch = []
        
        if self.count < batchSize:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batchSize)
            
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        
        return s_batch, a_batch, r_batch, t_batch, s2_batch
    
    def clear(self):
        self.deque.clear()
        self.count = 0