
import numpy as np
import tensorflow as tf
import tflearn






class SignalAgent(object):
    
    def __init__(self, sess, state_dim, action_dim, alpha, epsilon):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.alpha = alpha
        self.epsilon = 1
        self.epsilon_fin = epsilon
        
        ## Create networks
        self.inputs, self.out = self.create_network()
        
        ## Optimization ops
        self.target_Q = tf.placeholder(tf.float32, [None, self.a_dim])
        self.loss = tflearn.mean_square(self.out, self.target_Q)
        self.train_op = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        

        
    def create_network(self):
        inputs = tflearn.input_data(shape = [None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation = 'relu')
        net = tflearn.fully_connected(net, 300, activation = 'relu')
        
        w_init = tflearn.initializations.uniform(minval = -0.01, maxval = 0.01)
        
        out = tflearn.fully_connected(net, self.a_dim, activation = 'relu', weights_init = w_init)
        
        return inputs, out
    
    
    def chooseAction(self, state_):
        self.epsilon = max(self.epsilon - (0.01/30), self.epsilon_fin)
        
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(range(self.a_dim))
        else:
            return np.argmax(self.predict(state_))
        
        
    def predict(self, state_):
        return self.sess.run(self.out, feed_dict = {self.inputs : state_})
    
    def train(self, state_, target_):
        return self.sess.run(self.train_op, feed_dict = {self.inputs : state_, self.target_Q : target_})
        