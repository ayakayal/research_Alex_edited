import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random
import scipy.misc
#import tensorflow as tf
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import tf_slim as slim
from tensorflow import keras
from collections import defaultdict
class Qnetwork():
    def __init__(self ,num_states,h_size, num_actions, lr, scope):
        with tf.variable_scope(scope):
            #neuralnetwork option 1
            self.observation_input = keras.Input(shape=(num_states, ))
            self.dense1 = keras.layers.Dense(h_size, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(self.observation_input)
            self.dense2 = keras.layers.Dense(h_size, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')( self.dense1)
            self.q_out= keras.layers.Dense(
            num_actions, activation='linear', use_bias=True, kernel_initializer='glorot_uniform')(self.dense2)

            self.predict = tf.argmax(self.q_out,1)

            # Below we obtain the loss by taking the sum of squares difference 
            # between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,num_actions,dtype=tf.float32)
            
            self.Q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_onehot), axis=1)

            # Task 3: Compute the TD error
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
            self.update = self.trainer.minimize(self.loss)
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
class DQN_Agent:
    def __init__(self, env):
        self.env = env
        self.num_actions=env.actions_n
        self.env_solved = False
        self.threshold = 10
        self.state_space = len(env.states_n)
        self.action_space = env.actions_n

        self.intrinsic_reward = 1
        self.state_freq = {x: 0 for x in self.env.states}
        self.state_action_freq = defaultdict(lambda:1)
        self.score = 0
        self.intrinsic_score = 0
        self.epsilon=[]
 
    def train(self,num_episodes):
        #hyperparamters
        batch_size = 64 # How many experiences to use for each training step.
        update_freq = 4 # How often to perform a training step.
        y = .99 # Discount factor on the target Q-values
        startE = 1 # Starting chance of random action
        endE = 0.1 # Final chance of random action
        #anneling_steps = num_episodes * self.env.max_episode_steps #3000 # How many steps of training to reduce startE to endE.
        anneling_episodes= num_episodes
        num_episodes = num_episodes #000 #1000 How many episodes of game environment to train network with.
        pre_train_steps = 50 #500 # How many steps of random actions before training begins.
        model_path = "./models4/dqn" # The path to save our model to.
        summary_path = './summaries4/dqn' # The path to save summary statistics to.
        h_size = 256 # The number of units in the hidden layer.
        learning_rate = 1e-3 # Agent Learning Rate
        load_model = False # Whether to load a saved model.
        train_model = True # Whether to train the model
        self.scores = np.zeros(num_episodes)
        self.end_train=num_episodes #i added this if we want to stop the training after solving the game
# We get the shape of a state and the actions space size
#state_size = env.observation_space.shape[0]
        action_space_size = self.env.actions_n 
        # Number of episodes to run

        tf.reset_default_graph()
        mainQN = Qnetwork(len(self.env.observation_space.sample()),h_size, action_space_size, learning_rate, "main")
        #print('state space size',len(env.observation_space.sample()))
        #print('action space size',action_space_size)
        targetQN = Qnetwork(len(self.env.observation_space.sample()), h_size, action_space_size, learning_rate, "target")

        init = tf.global_variables_initializer()

        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
            
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver = tf.train.Saver()

        trainables = tf.trainable_variables()

        update_target_ops = update_target_graph("main", "target")

        myBuffer = experience_buffer()

        # Set the rate of random action decrease. 
        e = startE
        #stepDrop = (startE - endE)/anneling_steps
        stepDrop = (startE - endE)/anneling_episodes
        # Create lists to contain total rewards and steps per episode
        episode_lengths = []
        episode_rewards = []
        losses = []
        total_steps = 0

        with tf.Session() as sess:
            print('hey')
            sess.run(init)
            if load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
            for i in range(num_episodes):
                #time.sleep(0.3)
                episodeBuffer = experience_buffer()
                observation = self.env.reset()
                #observation = np.concatenate([observations, observations, observations], axis=2)
                done = False
                episode_reward = 0
                episode_steps = 0
        
                
                while not done and episode_steps< self.env.max_episode_steps:
                    episode_steps+=1
                    self.epsilon.append(e)
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if (np.random.rand(1) < e or total_steps < pre_train_steps) and train_model:
                        action = np.random.randint(0,action_space_size)
                    else:
                        action = sess.run(mainQN.predict, 
                                        feed_dict={mainQN.observation_input:[observation]})[0]
                        
                    if not train_model and np.random.rand(1) < 0.1:
                        action = np.random.randint(0,action_space_size)
                    observation_1, reward, done, _= self.env.step(self.env.actions[action])
                                            
                    total_steps += 1
                    
                    # Save the experience to our episode buffer.
                    episodeBuffer.add(np.reshape(np.array([observation,action,reward,observation_1,done]),[1,5])) 
                    #print('experience ',np.reshape(np.array([observation,action,reward,observation_1,done]),[1,5]))
                    if total_steps > pre_train_steps and train_model:
                        if total_steps % 1000 == 0: #It was 1000 steps
                            sess.run(update_target_ops)
                        #uncomment if you want anneling steps
                        # if e > endE:
                        #     e -= stepDrop
                        
                        if total_steps % (update_freq) == 0 and len(myBuffer.buffer)>=batch_size:
                            # Get a random batch of experiences.
                            trainBatch = myBuffer.sample(batch_size) 
                            # Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, 
                                        feed_dict={mainQN.observation_input: np.stack(trainBatch[:,3], axis=0)})
                            Q2 = sess.run(targetQN.q_out, 
                                        feed_dict={targetQN.observation_input: np.stack(trainBatch[:,3], axis=0)})
                            end_multiplier = -(trainBatch[:,4] - 1)
                            doubleQ = Q2[range(batch_size),Q1]
                            targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                            # Update the network with our target values.
                            _, q_loss = sess.run([mainQN.update, mainQN.loss],
                                feed_dict={mainQN.observation_input:np.stack(trainBatch[:,0], axis=0),
                                        mainQN.targetQ:targetQ, 
                                        mainQN.actions:trainBatch[:,1]})
                            losses.append(q_loss)
                    episode_reward += reward
                    self.score=episode_reward
                    observation = observation_1
                    
                
                            
                myBuffer.add(episodeBuffer.buffer)
                if e > endE:
                            e -= stepDrop
                episode_lengths.append(episode_steps)
                episode_rewards.append(episode_reward)
                self.scores[i]=episode_reward
                if i > self.threshold and np.all(self.scores[i-self.threshold: i] == self.env.total_reward):
                    self.env_solved = True
                   
                # Periodically save the model 
                if i % 1000 == 0 and i != 0:
                    saver.save(sess, model_path+'/model-'+str(i)+'.cptk')
                    print("Saved Model")
                if i % 10 == 0 and i != 0:
                    print ("Mean Reward: {}".format(np.mean(episode_rewards[-50:])))

