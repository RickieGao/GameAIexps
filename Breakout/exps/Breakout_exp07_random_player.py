import random
import breakout as game
import numpy as np

GAME = 'breakout'  # the name of the game being played for log files
ACTIONS = 3  # number of valid actions
PROBABILITY = 1  # probability of human choose
FRAME_PER_ACTION = 1
RUNNING = 4000000

def running():

	game_state = game.Main()
	t = 0
	episode = 0
	total_reward = 0
	reward_array = []
	max_q_array = []
	time_line_q = []
	time_line_r = []

	# get the first state by doing nothing
	do_nothing = np.zeros(ACTIONS)
	do_nothing[1] = 1
	x_t, r_0, terminal, ball_x, bat_mid = game_state.frame_step(do_nothing)

	while t <= RUNNING:
		a_t = np.zeros([ACTIONS])
		action_index = 1
		if t % FRAME_PER_ACTION == 0:
			# choosing the human action with PROBABILITY
			if random.random() <= PROBABILITY:
				print("----------Human Action----------")
				if ball_x < bat_mid:
					a_t = [1, 0, 0]  # move to left
				elif ball_x > bat_mid:
					a_t = [0, 0, 1]  # move to right
				else:
					a_t = [0, 1, 0]  # do nothing
			else :
				print("----------Random Action----------")
				action_index = random.randrange(ACTIONS)
				a_t[random.randrange(ACTIONS)] = 1
		else:
			a_t[1] = 1  # do nothing




