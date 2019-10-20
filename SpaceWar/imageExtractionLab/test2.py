import matplotlib.pyplot as plt
import math
# 初始学习率
learning_rate = 1
# 衰减系数
decay_rate = 0.8
# decay_steps控制衰减速度
# 如果decay_steps大一些,(global_step / decay_steps)就会增长缓慢一些
#   从而指数衰减学习率decayed_learning_rate就会衰减得慢一些
#   否则学习率很快就会衰减为趋近于0
decay_steps = 30000
# 迭代轮数
global_steps = 3000000
X = []
Y = []
# 指数学习率衰减过程
for global_step in range(1, global_steps):
    decayed_learning_rate = learning_rate * decay_rate**(global_step / decay_steps)
    # decayed_learning_rate = learning_rate * math.exp(-decay_rate / global_step)
    X.append(global_step / decay_steps)
    Y.append(decayed_learning_rate)
    # print("global step: %d, learning rate: %f" % (global_step, decayed_learning_rate))

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
curve = ax.plot(X,Y,'r',label="rule rate")
ax.legend()
ax.set_xlabel("t / decay_steps")
ax.set_ylabel("rule_rate")
plt.show()
