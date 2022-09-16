
import gym
import time
import math
import random
import itertools
import numpy as np 
import tensorflow as tf 
from statistics import mean
from collections import deque, namedtuple
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
from tensorflow.keras.callbacks import TensorBoard
from collections import defaultdict

summary_writer = tf.summary.create_file_writer(logdir = 'logs/dqn.txt')
class Model(tf.keras.Model):
	"""
	Subclassing a multi-layered NN using Keras from Tensorflow
	"""
	def __init__(self, num_states, hidden_units, num_actions):
		super(Model, self).__init__() # Used to run the init method of the parent class
		self.input_layer = kl.InputLayer(input_shape = (num_states,))
		self.hidden_layers = []

		for hidden_unit in hidden_units:
			self.hidden_layers.append(kl.Dense(hidden_unit,kernel_initializer='random_uniform', activation = 'relu')) # Left kernel initializer
		
		self.output_layer = kl.Dense(num_actions,kernel_initializer='random_uniform', activation = 'linear')
		#print('input shape ',(num_states,))
		

	@tf.function
	def call(self, inputs, **kwargs):
		x = self.input_layer(inputs)
		for layer in self.hidden_layers:
			x = layer(x)
			#print('X ',x)
		output = self.output_layer(x)
		return output

class ReplayMemory():
	"""
	Used to store the experience genrated by the agent over time
	"""
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.push_count = 0

	def push(self, experience):
		if len(self.memory)<self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.push_count % self.capacity] = experience
		self.push_count += 1

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
	"""
	Decaying Epsilon-greedy strategy
	"""
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_exploration_rate(self, current_step):
		return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)

def copy_weights(Copy_from, Copy_to):
	"""
	Function to copy weights of a model to other
	"""
	variables2 = Copy_from.trainable_variables
	variables1 = Copy_to.trainable_variables
	for v1, v2 in zip(variables1, variables2):
		v1.assign(v2.numpy())

class DQN_Agent:
	"""
	Used to take actions by using the Model and given strategy.
	"""
	def __init__(self, env):
		self.current_step = 0
		self.env = env
		self.num_actions=env.actions_n
		self.env_solved = False
		self.threshold = 10
		self.state_space = len(env.states_n)
		self.action_space = env.actions_n
		self.gamma = 0.99
		self.intrinsic_reward = 1
		self.state_freq = {x: 0 for x in self.env.states}
		self.state_action_freq = defaultdict(lambda:1)
		self.score = 0
		self.intrinsic_score = 0
		self.batch_size = 64
		self.eps_start = 1
		self.eps_end = 0.000
		self.eps_decay = 0.001
		self.target_update = 25
		self.memory_size = 100000
		self.lr = 0.01
		self.hidden_units = [200,200]
		self.strategy= EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)

	def select_action(self, state, policy_net):
		rate = self.strategy.get_exploration_rate(self.current_step)
		self.current_step += 1

		if rate > random.random():
			action_idx=random.randrange(self.num_actions)
			#print('action idx',action_idx)
			#print('action ',self.env.actions[action_idx])
			return action_idx,self.env.actions[action_idx], rate, True
		else:
			action_idx=np.argmax(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32'))))
			#print('khara ',policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32'))))
			#print('action idx',action_idx)
			#print('action ',self.env.actions[action_idx])
			return action_idx, self.env.actions[action_idx], rate, False


	""" 
	Notice that we are not using any function to make the states discrete here as DQN 
	can handle discrete state spaces.
	"""
	def train(self,epochs): #assign epochs to 10000 which is the number of episodes
	# Initialize Class variables
		strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)
		memory = ReplayMemory(self.memory_size)

	# Experience tuple variable to store the experience in a defined format
		Experience = namedtuple('Experience', ['states','actions', 'rewards', 'next_states', 'dones'])
	
	# Initialize the policy and target network
		policy_net = Model(len(self.env.observation_space.sample()), self.hidden_units, self.env.actions_n)
		target_net = Model(len(self.env.observation_space.sample()), self.hidden_units, self.env.actions_n)
	
	# Copy weights of policy network to target network
		copy_weights(policy_net, target_net)
		optimizer = tf.optimizers.Adam(self.lr)
		self.scores = np.zeros(epochs) # scores for each episode
		self.total_losses = np.zeros(epochs)
		self.intrinsic_scores = np.zeros(epochs)
		for epoch in range(epochs):
			#print('epoch ',epoch)
			state = self.env.reset()
			self.state_freq[state] += 1
			ep_rewards = 0 #rewards collected during 1 episode which is equivalent to score
			self.losses = []
			self.timestep=0

			while  self.timestep<self.env.max_episode_steps: # fix timesteps later 
				# Take action and observe next_stae, reward and done signal
				self.timestep+=1
				action_idx, action, rate, flag = self.select_action(state, policy_net)
				next_state, reward, done, _ = self.env.step(action)
				self.state_freq[next_state] += 1
				
	
				ep_rewards += reward
				#print('experience ',(state, action,action_idx, next_state, reward))
				# Store the experience in Replay memory
				memory.push(Experience(state, action_idx, next_state, reward, done)) #change to action_idx
			
				state = next_state

				if memory.can_provide_sample(self.batch_size):
					# Sample a random batch of experience from memory
					experiences = memory.sample(self.batch_size)
					batch = Experience(*zip(*experiences))

					# batch is a list of tuples, converting to numpy array here
					states, actions, rewards, next_states, dones = np.asarray(batch[0]),np.asarray(batch[1]),np.asarray(batch[3]),np.asarray(batch[2]),np.asarray(batch[4])
					
					# Calculate TD-target
					#print('output of target ',target_net(np.atleast_2d(next_states).astype('float32')))
					q_s_a_prime = np.max(target_net(np.atleast_2d(next_states).astype('float32')), axis = 1)
					q_s_a_target = np.where(dones, rewards, rewards+self.gamma*q_s_a_prime)
					q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype = 'float32')		
				
					# Calculate Loss function and gradient values for gradient descent
					with tf.GradientTape() as tape:
						#print('input shape ',(np.atleast_2d(states).astype('float32')).shape)
						#print('input ',(np.atleast_2d(states).astype('float32')))
						#print('policy net ',policy_net(np.atleast_2d(states).astype('float32')))
						#print('actions ',actions)
						q_s_a = tf.math.reduce_sum(policy_net(np.atleast_2d(states).astype('float32')) * tf.one_hot(actions, self.env.actions_n), axis=1)
						#print('tf ',tf.one_hot(actions, self.env.actions_n))
						#print(' q ',q_s_a)
						loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

					# Update the policy network weights using ADAM
					variables = policy_net.trainable_variables
					gradients = tape.gradient(loss, variables)
					optimizer.apply_gradients(zip(gradients, variables))

					self.losses.append(loss.numpy())

				else:
					self.losses.append(0)

				# If it is time to update target network
				if self.timestep%self.target_update == 0:
					copy_weights(policy_net, target_net)
			
				if done:
					break

			self.scores[epoch] = ep_rewards
			self.total_losses[epoch] = mean(self.losses)
			self.intrinsic_scores[epoch] = self.intrinsic_score
			if epoch > self.threshold and np.all(self.scores[epoch-self.threshold: epoch] == self.env.total_reward):
				self.env_solved = True
			avg_rewards = self.scores[max(0, epoch - 100):(epoch + 1)].mean() # Running average reward of 100 iterations
			
		# 	# Good old book-keeping
			with summary_writer.as_default():
				tf.summary.scalar('Episode_reward', self.scores[epoch], step = epoch)
				tf.summary.scalar('Running_avg_reward', avg_rewards, step = epoch)
				tf.summary.scalar('Losses', mean(self.losses), step = epoch)

		# 	if epoch%1 == 0:
		# 		print(f"Episode:{epoch} Episode_Reward:{total_rewards[epoch]} Avg_Reward:{avg_rewards: 0.1f} Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")

		# env.close()
	@staticmethod
	def reward_calc(base_reward, freq, t, alg='UCB'):
		if alg == 'UCB':
			return base_reward * np.sqrt(2 * np.log(t) / freq)
		if alg == 'MBIE-EB':
			return base_reward * np.sqrt(1 / freq)
		if alg == 'BEB':
			return base_reward / freq
		if alg == 'BEB-SQ':
			return base_reward / freq**2