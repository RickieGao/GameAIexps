import gym
import tensorflow as tf 
import numpy as np 
import random
from collections import deque
# from pyglet.gl import *
# gl_lib = pyglet.lib.load_library('GL')

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
LEARNING_RATE = 0.0001


class DQN:
	# DQN Agent
	def __init__(self, env):
		# init experience replay
		self.replay_buffer = deque()
		# init some parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n
		self.episode_reward = 0

		self.create_Q_network()
		self.create_training_method()

		# Init session
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())

		# loading networks
		self.saver = tf.train.Saver()

		# record numbers
		self.episode_reward = 0
		self.episode = 0
		self.reward_array = []
		self.max_q_array = []
		self.episode_array = []
		self.time_line = []

		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.session, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")

		global summary_writer
		summary_writer = tf.summary.FileWriter('results/Exp01Graph', graph=self.session.graph)

	def create_Q_network(self):
		# network weights
		W1 = self.weight_variable([self.state_dim, 20])
		b1 = self.bias_variable([20])
		W2 = self.weight_variable([20, self.action_dim])
		b2 = self.bias_variable([self.action_dim])
		# input layer
		self.state_input = tf.placeholder("float", [None, self.state_dim])
		# hidden layers
		h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
		# Q Value layer
		self.Q_value = tf.matmul(h_layer, W2) + b2

	def create_training_method(self):
		self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
		self.y_input = tf.placeholder("float", [None])
		Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		tf.summary.scalar("loss", self.cost)

		# record the reward
		with tf.name_scope('reward_per_life'):
			self.reward_record = tf.Variable(0.0, name='reward')
			self.reward_sum = tf.summary.scalar('reward_per_life', self.reward_record)

		# record the reward every 1000 time steps
		with tf.name_scope('reward_per_100_steps'):
			self.reward_step = tf.Variable(0.0, name='reward_step')
			self.reward_sum_step = tf.summary.scalar('reward_per_100 steps', self.reward_step)

		# placeholder to record reward
		self.reward_count = tf.placeholder('float')
		zero = tf.Variable(0.0, name='zero')
		self.re_count = 0.0
		self.life_count = 1
		self.reward_update = tf.assign(self.reward_record, self.reward_record + self.reward_count)
		self.reward_fresh = tf.assign(self.reward_record, zero)

		# placeholder to record reward_step
		self.reward_count_step = tf.placeholder('float')
		self.re_count_step = 0.0
		self.reward_update_step = tf.assign(self.reward_step, self.reward_step + self.reward_count_step)
		self.reward_fresh_step = tf.assign(self.reward_step, zero)

		global merged_summary_op
		merged_summary_op = tf.summary.merge_all()
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

	def perceive(self, state, action, reward, next_state, done):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()

		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Q_network(reward, done)

	def train_Q_network(self, reward, done):
		self.time_step += 1
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		# Step 2: calculate y
		y_batch = []
		Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
		for i in range(0, BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

		self.optimizer.run(feed_dict={
			self.y_input: y_batch,
			self.action_input: action_batch,
			self.state_input: state_batch
			})

		# record data
		if not done:
			self.episode_reward += reward
		else:
			self.reward_array.append(self.episode_reward)
			self.episode += 1
			self.episode_array.append(self.episode)
			self.episode_reward = 0


		# self.re_count += reward
		# self.re_count_step += reward
		# self.episode_reward += reward
		# if done:
		# 	self.session.run(self.reward_update, feed_dict={self.reward_count: float(self.re_count)})
		# 	re = self.session.run(self.reward_sum)
		# 	summary_writer.add_summary(re, self.life_count)
		# 	self.session.run(self.reward_fresh)
		# 	self.re_count = 0
		# 	self.life_count += 1
		# 	self.episode_reward = 0
		#
		# if self.time_step % 100 == 0:
		# 	self.session.run(self.reward_update_step, feed_dict={self.reward_count_step: float(self.re_count_step)})
		# 	re_step = self.session.run(self.reward_sum_step)
		# 	summary_writer.add_summary(re_step, self.time_step * 100)
		# 	self.session.run(self.reward_fresh_step)
		# 	self.re_count_step = 0
		summary_str = self.session.run(merged_summary_op, feed_dict={
				self.y_input: y_batch,
				self.action_input: action_batch,
				self.state_input: state_batch
				})
		summary_writer.add_summary(summary_str, self.time_step)

		# save network every 1000 iteration
		if self.time_step % 1000 == 0:
			self.saver.save(self.session, 'results/saved_networks/' + 'network' + '-dqn', global_step=self.time_step)

	def egreedy_action(self, state):
		Q_value = self.Q_value.eval(feed_dict={
			self.state_input: [state]
			})[0]
		if random.random() <= self.epsilon:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
			action = random.randint(0, self.action_dim - 1)
		else:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
			action = np.argmax(Q_value)
		self.max_q_array.append(np.max(Q_value))
		self.time_line.append(self.time_step)
		return action

	def action(self, state):
		return np.argmax(self.Q_value.eval(feed_dict = {
			self.state_input: [state]
			})[0])

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape=shape)
		return tf.Variable(initial)


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
	# initialize OpenAI Gym env and dqn agent
	env = gym.make(ENV_NAME)
	agent = DQN(env)
	# initialize task
	state = env.reset()
	episode_reward = 0
	while agent.time_step <= 100000:
		# Train
		env.render()
		action = agent.egreedy_action(state)  # e-greedy action for train
		next_state, reward, done, _ = env.step(action)
		# Define reward for agent
		reward_agent = -1 if done else 0.1
		agent.perceive(state, action, reward_agent, next_state, done)
		state = next_state

		print("TIMESTEP", int(agent.time_step), "/ EPSILON", agent.epsilon, "/ ACTION", action, "/ REWARD", reward_agent,
				"/ EPISODE_REWARD", agent.episode_reward, "/ TERMINAL", done)
		if done:
			env.reset()

	# restore lists
	time_line_r_file = open(r'results\exp01dqn\episodes.txt', 'w')
	for word in agent.episode_array:
		time_line_r_file.write(str(word))
		time_line_r_file.write('\n')
	time_line_r_file.close()

	time_line_q_file = open(r'results\exp01dqn\time_line_q.txt', 'w')
	for word in agent.time_line:
		time_line_q_file.write(str(word))
		time_line_q_file.write('\n')
	time_line_q_file.close()

	reward_array_file = open(r'results\exp01dqn\reward_array.txt', 'w')
	for word in agent.reward_array:
		reward_array_file.write(str(word))
		reward_array_file.write('\n')
	reward_array_file.close()

	max_q_array_file = open(r'results\exp01dqn\max_q_array.txt', 'w')
	for word in agent.max_q_array:
		max_q_array_file.write(str(word))
		max_q_array_file.write('\n')
	max_q_array_file.close()
	print("finished!")


if __name__ == '__main__':
	main()
