from matplotlib import pyplot as plt

# 初始学习率
learning_rate = 0.8
# 衰减系数
decay_rate = 0.8
# decay_steps控制衰减速度
# 如果decay_steps大一些,(global_step / decay_steps)就会增长缓慢一些
#   从而指数衰减学习率decayed_learning_rate就会衰减得慢一些
#   否则学习率很快就会衰减为趋近于0
decay_steps = 30000
# 迭代轮数
global_steps = 1000000

X = []
Y = []
# 指数学习率衰减过程
for global_step in range(global_steps):
    decayed_learning_rate = learning_rate * decay_rate**(global_step / decay_steps)
    X.append(global_step)
    Y.append(decayed_learning_rate)
    # print("global step: %d, learning rate: %f" % (global_step,decayed_learning_rate))

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

plt.figure()
plt.title("生效概率随训练时间指数下降", fontsize=16)
plt.xlabel("训练时间", fontsize=16)
plt.ylabel("生效概率", fontsize=16)
plt.grid(ls='--')
dqn, = plt.plot(X, Y, color="#cc3311")
plt.show()

