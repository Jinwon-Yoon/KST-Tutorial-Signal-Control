

##### Importing Moduels #####
%matplotlib inline

from TrafficEnv import *
from SignalAgent import * 
from ReplayBuffer import *

import numpy as np
import matplotlib.pyplot as plt

################################



##### Hyper Parameters #######
GAMMA = 0.95
ALPHA = 0.001
EPSILON = 0.1

MAX_EPISODES = 300
MAX_EP_STEPS = 30

BUFFER_SIZE = 30000
BATCH_SIZE = 32
RANDOM_SEED = 1234

avgRange = 10
rewards = []

################################



##### Proceed the episode & Update the model #####
# sess = tf.Session()
with tf.Session() as sess:
    
    state_dim = 4
    action_dim = 4
    
    env = TrafficEnv()
    agent = SignalAgent(sess, state_dim, action_dim, ALPHA, EPSILON)
    replayBuffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    sess.run(tf.global_variables_initializer())    
        
    
    for i in range(MAX_EPISODES):
        
        ##### Initialization #####
        env.startSUMO()
        currentState = env.reset()
        
        accumulatedRewards = 0
        
        
        for t in range(MAX_EP_STEPS):
            
            ##### Action Selection #####
            currentAction_idx  = agent.chooseAction(np.reshape(currentState, (1, 4)))
            currentAction = (currentAction_idx + 1) * 0.2
            
            
            ##### Step the simulation & Observe the experience #####
            nextState, reward, terminal = env.step(currentAction)
            replayBuffer.add(np.reshape(currentState, (agent.s_dim, )), currentAction, 
                             reward, terminal, np.reshape(nextState, (agent.s_dim, )))
            
            
            ##### Batch Learning #####
            if replayBuffer.size() > BATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replayBuffer.sampleBatch(BATCH_SIZE)
                
                currentQ = agent.predict(s_batch)
                newQ = agent.predict(s2_batch)
                
                y = np.copy(currentQ)
                for k in range(BATCH_SIZE):
                    a_idx = int((a_batch[k] / 0.2) - 1)
                    if t_batch[k]:
                        y[k, a_idx] = r_batch[k]
                    else:
                        y[k, a_idx] = r_batch[k] + GAMMA * max(newQ[k, :])
                        
                agent.train(s_batch, y)
                
                
            ##### proceeding the timestep #####   
            currentState = nextState
            accumulatedRewards += reward
        
        
        rewards.append(accumulatedRewards)
        
        env.endSUMO()
        env.timestep = -1
        print("[Episode : ", i, "]  Rewards : %.4f" % accumulatedRewards, "Epsilon : %.3f" % agent.epsilon)




##### Draw the graph to test the convergence #####
smoothedRewards = np.copy(rewards)
for i in range(avgRange, MAX_EP_STEPS):
    smoothedRewards[i] = np.mean(rewards[i - avgRange : i + 1])
    
plt.figure(1)
plt.plot(smoothedRewards, label = 'DQN control')
plt.xlabel('Episodes')
plt.ylabel('Accumulated Rewards')
plt.legend()
##################################################
