from matplotlib import pyplot as plt
X = []
Y = []
learning_rate = 0.9
decay_rate = 0.4
decay_steps = 10
# 指数学习率衰减过程
global_steps = 140
for global_step in range(global_steps):
    decayed_learning_rate = learning_rate * decay_rate**(global_step / decay_steps)
    X.append(global_step)
    Y.append(decayed_learning_rate)
    # print("global step: %d, learning rate: %f" % (global_step, decayed_learning_rate))

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
curve = ax.plot(X,Y,'r',label="omega")
ax.legend()
ax.set_xlabel("global_step / decay_steps")
ax.set_ylabel("omega")
plt.show()