import csv
from decimal import Decimal
import matplotlib
from matplotlib import pyplot as plt

WEIGHT_R = 0.98
WEIGHT_Q = 0.6


def read_file(file_name):
	with open(file_name, "r") as file:
		reader = csv.reader(file)
		next(reader)
		recorder1, recorder2 = [], []
		for row in reader:
			recorder1.append(int(row[1]))
			recorder2.append(float(Decimal(row[2]).quantize(Decimal('0.00'))))
	return recorder1, recorder2


def smooth(target, wight):
	smoothed = []
	last = target[0]
	for value in target:
		smoothed_val = last * wight + (1 - wight) * value
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed


DQN_timesteps, DQN_Max_Q = read_file(r"bird\DQN-tag-50kper_average_qMax_50kper_average_qMax.csv")
rule_timesteps, rule_Max_Q = read_file(r"bird\run-DQN_rule-tag-50kper_average_qMax_50kper_average_qMax.csv")
DQN_episode, DQN_reward = read_file(r"bird\run-DQN-tag-reward_per_life_reward_per_life.csv")
rule_episode, rule_reward = read_file(r"bird\run-DQN_rule-tag-reward_per_life_reward_per_life.csv")

reve_timesteps, reve_Max_Q = read_file(r"bird\run-DQN_rule_reve_count_graph-tag-50kper_average_qMax_50kper_average_qMax.csv")
reve_episode, reve_reward = read_file(r"bird\run-DQN_rule_reve_count_graph-tag-reward_per_life_reward_per_life.csv")

DQN_timesteps = DQN_timesteps[:len(rule_timesteps)]
DQN_Max_Q = DQN_Max_Q[:len(rule_Max_Q)]

reve_timesteps = reve_timesteps[:len(rule_timesteps)]
reve_Max_Q = reve_Max_Q[:len(rule_Max_Q)]

DQN_episode = DQN_episode[:len(DQN_episode) // 2]
DQN_reward = DQN_reward[: len(DQN_reward) // 2]
rule_episode = rule_episode[: len(rule_reward) // 2]
rule_reward = rule_reward[: len(rule_reward) // 2]
reve_episode = reve_episode[: len(reve_reward) // 2]
reve_reward = reve_reward[: len(reve_reward) // 2]

DQN_timesteps = DQN_timesteps[:len(DQN_timesteps) // 2]
DQN_Max_Q = DQN_Max_Q[: len(DQN_Max_Q) // 2]
rule_timesteps = rule_timesteps[: len(rule_timesteps) // 2]
rule_Max_Q = rule_Max_Q[: len(rule_Max_Q) // 2]
reve_timesteps = reve_timesteps[: len(reve_timesteps) // 2]
reve_Max_Q = reve_Max_Q[: len(reve_Max_Q) // 2]

smoothed_DQN_Max_Q = smooth(DQN_Max_Q, WEIGHT_Q)
smoothed_rule_Max_Q = smooth(rule_Max_Q, WEIGHT_Q)
smoothed_DQN_reward = smooth(DQN_reward, WEIGHT_R)
smoothed_rule_reward = smooth(rule_reward, WEIGHT_R)

smoothed_reve_Max_Q = smooth(reve_Max_Q, WEIGHT_Q)
smoothed_reve_reward = smooth(reve_reward, WEIGHT_R)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.figure()
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Average Reward", fontsize=20)
plt.title("Average Reward on Flappy Bird", fontsize=20)
dqn, = plt.plot(DQN_episode, smoothed_DQN_reward, color="#cc3311")
dqn_rule, = plt.plot(rule_episode, smoothed_rule_reward, color="#0077bb")
dqn_reve, = plt.plot(reve_episode, smoothed_reve_reward, color='#33E6cc')
# plt.legend(handles=[dqn, dqn_rule], labels=['DQN', 'RIL'],  loc='upper left')
plt.legend(handles=[dqn, dqn_rule, dqn_reve], labels=['DQN', 'RIL', 'RIL with reverse rules'],  loc='upper left')
# plt.axvline(x=26950, color='black', linestyle="--")
# plt.axvline(x=29000, color='black', linestyle="--")
# plt.axhline(y=70, color='black', linestyle="--")
# plt.axhline(y=44, color='black', linestyle="--")
plt.figure()
plt.title("Average Q on Flappy Bird", fontsize=20)
plt.xlabel("Training Epochs", fontsize=20)
plt.ylabel("Average Q Value", fontsize=20)
dqn_q, = plt.plot(DQN_timesteps, smoothed_DQN_Max_Q, color="#cc3311")
rule_q, = plt.plot(rule_timesteps, smoothed_rule_Max_Q, color="#0077bb")
reve_q, = plt.plot(reve_timesteps, smoothed_reve_Max_Q, color='#33e6cc')
plt.legend(handles=[dqn_q, rule_q, reve_q], labels=['DQN', 'RIL', 'RIL with reverse rules'],  loc='lower right')
plt.show()
