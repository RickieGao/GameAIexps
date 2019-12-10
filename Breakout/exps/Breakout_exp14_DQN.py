import tensorflow as tf
import numpy as np
import cv2
import sys
import random
from collections import deque
import pygame
# import copy
# import gym
sys.path.append("game/")
import breakout_2actions as game
# from image_processing import *  # functions to process images (game frames)


GAME = 'breakout'  # the name of the game being played for log files
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 1000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
LEARNING_RATE = 1e-6
# INITIAL_OMEGA = 0.5  # OMEGA is the probability of rule
# FINAL_OMEGA = 0
GAMMA = 0.99  # decay rate of past observations
ACTIONS = 2  # number of valid actions
FRAME_PER_ACTION = 1


def setRandomSeed(seed):
    # eliminate random factors
    random.seed(seed)
    tf.set_random_seed(seed)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# insert tensorboard recorder for each network layer
def createNetwork():
    # input layer
    with tf.name_scope('input'):
        # 80 * 80 * 4 image
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

    return s, readout, W_fc1, W_fc2  # s: the placeholder of input. readout: the network final function.


def trainNetwork(s, readout, W_fc1, W_fc2, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS], name='action')
    y = tf.placeholder("float", [None], name='q_next')
    tf.summary.histogram('q_next', y)
    with tf.name_scope('q_eval'):
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        tf.summary.histogram('fc2/output', readout_action)
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.square(y - readout_action))
        tf.summary.scalar('loss', cost)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # network difference
    regularize_lambda = 1.0
    regularizer = tf.contrib.layers.l2_regularizer(regularize_lambda)  # equal to tf.nn.l2_loss

    with tf.name_scope('weight'):
        last_W_fc1 = tf.Variable(tf.constant(0.0, shape=W_fc1.get_shape()))
        diff_W_fc1 = tf.contrib.layers.apply_regularization(regularizer, [W_fc1 - last_W_fc1])
        tf.summary.scalar('diff_W_fc1', diff_W_fc1)
        last_W_fc1_update = tf.assign(last_W_fc1, W_fc1)

        last_W_fc2 = tf.Variable(tf.constant(0.0, shape=W_fc2.get_shape()))
        diff_W_fc2 = tf.contrib.layers.apply_regularization(regularizer, [W_fc2 - last_W_fc2])
        tf.summary.scalar('diff_W_fc2', diff_W_fc2)
        last_W_fc2_update = tf.assign(last_W_fc2, W_fc2)

    # tensorboard output
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(r"result/Exp8Graph/", sess.graph)

    # record the reward
    with tf.name_scope('reward_per_life'):
        reward = tf.Variable(0.0, name='reward')
        reward_sum = tf.summary.scalar('reward_per_life', reward)

    # record the reward every 1000 time steps
    with tf.name_scope('reward_per_10000_steps'):
        reward_step = tf.Variable(0.0, name='reward_step')
        reward_sum_step = tf.summary.scalar('reward_per_10000_steps', reward_step)

    # placeholder to record reward
    reward_count = tf.placeholder('float')
    zero = tf.Variable(0.0, name='zero')
    re_count = 0.0
    life_count = 1
    reward_update = tf.assign(reward, reward + reward_count)
    reward_fresh = tf.assign(reward, zero)

    # placeholder to record reward_step
    reward_count_step = tf.placeholder('float')
    re_count_step = 0.0
    reward_update_step = tf.assign(reward_step, reward_step + reward_count_step)
    reward_fresh_step = tf.assign(reward_step, zero)

    # record average max q value
    with tf.name_scope('50kper_average_qMax'):
        q_max = tf.Variable(0.0, name='q_max_average')
        q_max_sum = tf.summary.scalar('50kper_average_qMax', q_max)
    # placeholder to record q value
    q_max_count = tf.placeholder('float')
    q_count = 0.0
    batch_count = 0
    q_max_update = tf.assign(q_max, q_max_count)

    # open up a game state to communicate with emulator
    game_state = game.Main()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[1] = 1
    x_t_colored, r_0, terminal, _, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # variable to save pygame frame
    # game_frame = x_t_colored

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(r"result/Exp14_saved_networks")

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    # omega = INITIAL_OMEGA
    t = 0
    episode_reward = 0
    while True:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]  # returns output of current input(images).
        a_t = np.zeros([ACTIONS])
        action_index = 1
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[1] = 1  # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # scale down omega
        # if omega < FINAL_OMEGA and t > OBSERVE:
        #     omega -= (FINAL_OMEGA - INITIAL_OMEGA) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal, _, _ = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # update pygame frame
        # game_frame = x_t1_colored

        episode_reward += r_t

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)  # experience replay.

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})  # readout_j1_batch: Q value?

            # average max q value
            batch_count += 1
            q_count += np.max(readout_t)
            BATCH_N = 50000
            if batch_count % BATCH_N == 0:
                sess.run(q_max_update, feed_dict={q_max_count: float(q_count / BATCH_N)})
                qm = sess.run(q_max_sum)
                writer.add_summary(qm, batch_count)
                q_count = 0.0

            # total reward
            re_count += r_t
            re_count_step += r_t

            if terminal:
                sess.run(reward_update, feed_dict={reward_count: float(re_count)})
                re = sess.run(reward_sum)
                writer.add_summary(re, life_count)
                sess.run(reward_fresh)
                re_count = 0
                life_count += 1

            if t % 10000 == 0:
                sess.run(reward_update_step, feed_dict={reward_count_step: float(re_count_step)})
                re_step = sess.run(reward_sum_step)
                writer.add_summary(re_step, t * 10000)
                sess.run(reward_fresh_step)
                re_count_step = 0

            for i in range(0, len(minibatch)):
                ith_terminal = minibatch[i][4]
                # if terminal, only equals reward
                if ith_terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict={  # feed back to update network.
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

            # update tensorboard data per 1000 steps
            if t % 1000 == 0:
                result = sess.run(merged, feed_dict={
                    y: y_batch,
                    a: a_batch,
                    s: s_j_batch})
                writer.add_summary(result, t)

            # record network weight
            if t % 1000 == 0:
                sess.run([last_W_fc1_update, last_W_fc2_update])

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'result/Exp14_saved_networks/' + GAME + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ EPISODE_REWARD", episode_reward,
              "/ Q_MAX %e" % np.max(readout_t), "/ TERMINAL", terminal)
        if terminal:
            episode_reward = 0


def playGame():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    s, readout, W_fc1, W_fc2 = createNetwork()
    trainNetwork(s, readout, W_fc1, W_fc2, sess)


def main():
    setRandomSeed(11)
    pygame.init()
    playGame()


if __name__ == "__main__":
    main()
