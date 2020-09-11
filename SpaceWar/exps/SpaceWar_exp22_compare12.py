import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
sys.path.append("ruleset/")
import RuleAction2
import plane as game
import random
import numpy as np
from collections import deque
# from image_processing import *


ACTIONS = 3									# number of valid actions
GAMMA = 0.99								# decay rate of past observations
REPLAY_MEMORY = 50000						# number of previous transitions to remember
BATCH = 32									# size of minibatch
FRAME_PER_ACTION = 1
OBSERVE = 0
EXPLORE = 3000000
FINAL_EPSILON = 0
INITIAL_EPSILON = 0
LEARNING_RATE = 1e-6


def setRandomSeed(seed):
	# eliminate random factors
	random.seed(seed)
	tf.set_random_seed(seed)


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape)
	return tf.Variable(initial)


def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def createNetwork():
	# input layer
	with tf.name_scope('input'):
		s = tf.placeholder("float", [None, 80, 80, 4])

	# network weights
	with tf.name_scope('conv1_layer'):
		with tf.name_scope('weight_conv1'):
			W_conv1 = weight_variable([8, 8, 4, 32])
			tf.summary.histogram('conv1/weights', W_conv1)
		with tf.name_scope('bias_conv1'):
			b_conv1 = bias_variable([32])
			tf.summary.histogram('conv1/bias', b_conv1)
		h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
		tf.summary.histogram('conv1/output', h_conv1)
	
	with tf.name_scope('pool1_layer'):
		h_pool1 = max_pool_2x2(h_conv1)
		tf.summary.histogram('pool1/output', h_pool1)

	with tf.name_scope('conv2_layer'):
		with tf.name_scope('weight_conv2'):
			W_conv2 = weight_variable([4, 4, 32, 64])
			tf.summary.histogram('conv2/weights', W_conv2)
		with tf.name_scope('bias_conv2'):
			b_conv2 = bias_variable([64])
			tf.summary.histogram('conv2/bias', b_conv2)
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
		tf.summary.histogram('conv2/output', h_conv2)

	with tf.name_scope('conv3_layer'):
		with tf.name_scope('weight_conv3'):
			W_conv3 = weight_variable([3, 3, 64, 64])
			tf.summary.histogram('conv3/weights', W_conv3)
		with tf.name_scope('bias_conv3'):
			b_conv3 = bias_variable([64])
			tf.summary.histogram('conv3/bias', b_conv3)
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
		tf.summary.histogram('conv3/output', h_conv3)

	with tf.name_scope('reshape'):
		h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
		tf.summary.histogram('reshape/output', h_conv3_flat)

	with tf.name_scope('fc1_layer'):
		with tf.name_scope('weight_fc1'):
			W_fc1 = weight_variable([1600, 512])
			tf.summary.histogram('fc1/weights', W_fc1)
		with tf.name_scope('bias_fc1'):
			b_fc1 = bias_variable([512])
			tf.summary.histogram('fc1/bias', b_fc1)
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
		tf.summary.histogram('fc1/output', h_fc1)

	with tf.name_scope('fc2_layer'):
		with tf.name_scope('weight_fc2'):
			W_fc2 = weight_variable([512, ACTIONS])
			tf.summary.histogram('fc2/weights', W_fc2)
		with tf.name_scope('bias_fc2'):
			b_fc2 = bias_variable([ACTIONS])
			tf.summary.histogram('fc2/bias', b_fc2)
		# readout layer
		readout = tf.matmul(h_fc1, W_fc2) + b_fc2

	return s, readout, W_fc1, W_fc2


def trainNetwork(s, readout, W_fc1, W_fc2, sess):
	# open up a game state to communicate with emulator
	game_state = game.GameState()

	# get the first state by doing nothing and preprocess the image to 80x80x4
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1
	x_t_colored, r_0, terminal = game_state.frame_step(do_nothing)
	x_t = cv2.cvtColor(cv2.resize(x_t_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
	s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

	# variable to save pygame frame
	pygame_frame = x_t_colored

	# saving and loading networks
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	checkpoint = tf.train.get_checkpoint_state("saved_networks/Exp24_saved_networks")

	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old network weights")

	# start training
	epsilon = INITIAL_EPSILON
	t = 0
	rule_action_count = 0
	anti_action_count = 0
	stand_count = 0
	while t < 100001:
		# choose an action epsilon greedily
		readout_t = readout.eval(feed_dict={s: [s_t]})[0]
		a_t = np.zeros([ACTIONS])
		if t % FRAME_PER_ACTION == 0:
			if random.random() <= epsilon:
				print("----------Random Action----------")
				a_t[random.randrange(ACTIONS)] = 1
			else:
				action_index = np.argmax(readout_t)
				a_t[action_index] = 1
		else:
			a_t[0] = 1  # do nothing

		# run the selected action and observe next state and reward
		x_t1_colored, r_t, terminal = game_state.frame_step(a_t)				# returns new game images and new reward.
		x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
		x_t1 = np.reshape(x_t1, (80, 80, 1))
		s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

		# count rule action
		action_map = {'right': 'R', 'left': 'L'}
		anti_action_map = {'right': 'L', 'left': 'R'}
		anti_map = {(0, 1, 0): 'L', (0, 0, 1): 'R', (1, 0, 0): 'N'}
		model_action = anti_map[tuple(a_t.tolist())]
		rule_action = action_map[RuleAction2.rule_action(pygame_frame)]
		anti_rule_action = anti_action_map[RuleAction2.rule_action(pygame_frame)]
		# np.save("image_data//rule_input_image//rule_input" + str(t) + rule_action + model_action + ".npy", s_t)
		# np.save("image_data//rule_image//rule_" + str(t) + rule_action + model_action + ".npy", pygame_frame)
		if rule_action == model_action:
			rule_action_count += 1
		if anti_rule_action == model_action:
			anti_action_count += 1
		if model_action == 'N':
			stand_count += 1

		s_t = s_t1
		t += 1
		pygame_frame = x_t1_colored

		print("TIMESTEP", t, "/ REWARD", r_t, "/ RULE_ACTION_COUNT", rule_action_count,
				'/ ANTI_ACTION_COUNT', anti_action_count, "/ STAND_COUNT", stand_count)

	print('rule_action:', rule_action_count, 'anti_rule_action:', anti_action_count, 'stand_action:', stand_count)
	print('finished!')


def playGame():
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
	s, readout, W_fc1, W_fc2 = createNetwork()
	trainNetwork(s, readout, W_fc1, W_fc2, sess)


def main():
	setRandomSeed(11)
	playGame()


if __name__ == "__main__":
	main()
