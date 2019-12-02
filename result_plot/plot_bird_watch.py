from decimal import Decimal
from matplotlib import pyplot as plt


def read_file(file_name):
	with open(file_name, "r") as file:
		recorder = []
		for lines in file:
			recorder.append(float(Decimal(lines).quantize(Decimal('0.00'))))
	return recorder


def smooth(target, wight):
	smoothed = []
	last = target[0]
	for value in target:
		smoothed_val = last * wight + (1 - wight) * value
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed


DQN_time_line = read_file(r"breakout_watch\dqn_time_line.txt")
DQN_rate = read_file(r"breakout_watch\dqn_action_times_array.txt")

rule_time_line = read_file(r"breakout_watch\rule_time_line.txt")
rule_rate = read_file(r"breakout_watch\rule_action_times_array.txt")

reve_time_line = read_file(r"breakout_watch\reve_time_line.txt")
reve_rate = read_file(r"breakout_watch\reve_action_times_array.txt")


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
fig_r, ax_r = plt.subplots()
plt.xlabel("time", fontsize=20)
plt.ylabel("rule action times", fontsize=20)
# plt.title("Average Reward on Breakout", fontsize=20)
dqn, = plt.plot(DQN_time_line, DQN_rate, color="#cc3311")
dqn_rule, = plt.plot(rule_time_line, rule_rate, color="#0077bb")
dqn_reve, = plt.plot(reve_time_line, reve_rate, color="#33E6CC")
plt.legend(handles=[dqn, dqn_rule, dqn_reve], labels=['DQN', 'RIL', 'reverse rule'],  loc='lower right')
# plt.axvline(x=20750, color='black', linestyle="--")
# plt.axvline(x=30500, color='black', linestyle="--")
# plt.axhline(y=55, color='black', linestyle="--")
# plt.axhline(y=26.5, color='black', linestyle="--")
# fig_q, ax_q = plt.subplots()
# plt.xlabel("Training Epochs", fontsize=20)
# plt.ylabel("Average Q Value", fontsize=20)
# plt.title("Average Q on Breakout", fontsize=20)
# dqn_q, = plt.plot(time_line_q, smoothed_max_q, color="#cc3311")
# dqn_rule_q, = plt.plot(rule_time_line_q, smoothed_rule_max_q, color="#0077bb")
# plt.legend(handles=[dqn_q, dqn_rule_q], labels=['DQN', 'RIL'],  loc='lower right')
plt.show()
