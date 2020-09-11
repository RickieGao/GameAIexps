from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

# add , 'CartPole', 'Pong'
game = ['Flappy bird', 'Space war', 'Breakout']
score = [0.0793, 0.2, 0.3197]
score_percent = ["%.2f%%" % (a * 100) for a in score]
x = range(len(score))


def to_percent(temp, position):
    return '%.0f' % (100*temp) + '%'


# set the dpi
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

fig, ax = plt.subplots(figsize=(8, 5))

ax.set_xlabel('Improvement', color='k', fontsize=16)
ax.set_title('Improvement by Accelerate Rule Set', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.barh(x, score, tick_label=game, height=0.5, alpha=0.8)
ax.set_xlim([0, 0.4])
for x, y in enumerate(score):
    ax.text(y + 0.02, x, score_percent[x], va='center', fontsize=14)
plt.show()
