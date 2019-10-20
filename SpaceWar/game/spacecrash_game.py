# -*- coding: utf-8 -*-

import pygame
from myGameRole import *
# 导入按键的检测
from pygame.locals import *
import time
import random


class GameState:
	# pygame初始化
	# pygame.init()
	def __init__(self):
		# 创建窗口
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		# 读出背景图片
		self.background = pygame.image.load('resources/image/background.png').convert()

		# 创建对象
		self.playerPlane = Player(self.screen)

		# 敌机列表
		self.enemyList = []
		# 敌机产生频率
		self.enemy_frequency = 0

		self.shoot_down_enemy = False

	def frame_step(self, input_actions):
		# 1.显示背景
		self.screen.blit(self.background, (0, 0))
		self.isterminate = False
		self.reward = 0

		# 产生敌方飞机
		if self.enemy_frequency % 30 == 0:
			num = random.randint(0, SCREEN_WIDTH - 51)
			enemy = Enemy(self.screen, num)
			self.enemyList.append(enemy)
		self.enemy_frequency += 1
		if self.enemy_frequency >= 100:
			self.enemy_frequency = 0

		# plane move
		if sum(input_actions) != 1:
			raise ValueError('Multiple input actions!')

		if input_actions[0] == 1:
			self.playerPlane.keyHandle('up')
		if input_actions[1] == 1:
			self.playerPlane.keyHandle('down')
		if input_actions[2] == 1:
			self.playerPlane.keyHandle('left')
		if input_actions[3] == 1:
			self.playerPlane.keyHandle('right')
		if input_actions[4] == 1:
			self.playerPlane.keyHandle('space')
		# shooting all the time
		# self.playerPlane.keyHandle('space')

		# 这里只是用于检测敌机的状态，如果出界就直接移除，如果被子弹打中就执行爆炸效果随后与子弹一起移除
		# 还有一点，图片刷新不能与此同时进行，会有一顿一顿的感觉，所以只在执行爆炸效果时进行刷新，最后统一刷新
		for enemy in self.enemyList:
			enemy.move()

			if enemy.checkOut():
				self.enemyList.remove(enemy)
				continue
			
			x1, y1, w1, h1 = enemy.getPos()

			for bullet in self.playerPlane.bulletList:
				x2, y2, w2, h2 = bullet.getPos()

				if (x2 + w2 // 2) >= x1 and (x2 + w2 // 2) <= (x1 + w1):
					if y2 <= (y1 + h1):
						enemy.crash()
						self.playerPlane.bulletList.remove(bullet)
						self.enemyList.remove(enemy)
						enemy.draw()
						self.shoot_down_enemy = True
						# enemy_down_sound.play()
						break

			x3, y3, w3, h3 = self.playerPlane.getPos()
			if (x1 + w1 // 2) >= x3 and (x1 + w1 // 2) <= (x3 + w3):
				if y3 <= (y1 + h1):
					# playerPlane.crash()
					# # 更新飞机图片
					# playerPlane.draw()
					# running_control = False
					# break
					self.isterminate = True

		for enemy in self.enemyList:
			enemy.draw()

		# 子弹移动，更新图片
		for bullet in self.playerPlane.bulletList:
			bullet.move()
			bullet.draw()

		# 清楚发射到顶部的子弹
		self.playerPlane.bulletClear()

		# 如果结束，则重启游戏，并设置reward：游戏结束奖励为-3，打落敌机奖励为1，无事件发生则奖励为0
		if self.isterminate:
			self.playerPlane.crash()
			self.reward = -1
			self.__init__()
			# game_over_sound.play()
		elif self.shoot_down_enemy:
			self.reward = 1
			self.shoot_down_enemy = False
		else:
			self.reward = 0.1

		# 更新飞机图片
		self.playerPlane.draw()

		# 整个界面刷新
		pygame.display.update()

		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		# 延时0.01s
		# time.sleep(0.03)
		return image_data, self.reward, self.isterminate

	# 游戏结束，退出循环
	# while True:
	# 	# 读出背景图片
	# 	background = pygame.image.load('resources/image/gameover.png').convert()
	# 	screen.blit(background, (0, 0))
	# 	pygame.display.update()
	#
	# 	for event in pygame.event.get():
	# 		if event.type == QUIT:
	# 			print('exit')
	# 			exit()
	#
	# 	time.sleep(0.03)


def human_play():
	# pygame初始化
	pygame.init()
	game = GameState()

	while game.isterminate:
		# 判断按键
		for event in pygame.event.get():
			if event.type == QUIT:
				print('exit')
				exit()
			elif event.type == KEYDOWN:
				if event.key == K_a or event.key == K_LEFT:
					# print('left')
					# playerPlane.keyHandle('left')
					game.frame_step([0, 0, 1, 0])
				elif event.key == K_d or event.key == K_RIGHT:
					# print('right')
					# playerPlane.keyHandle('right')
					game.frame_step([0, 0, 0, 1])
				elif event.key == K_w or event.key == K_UP:
					# print('up')
					# playerPlane.keyHandle('up')
					game.frame_step([1, 0, 0, 0])
				elif event.key == K_s or event.key == K_DOWN:
					# print('down')
					# playerPlane.keyHandle('down')
					game.frame_step([0, 1, 0, 0])
				elif event.key == K_SPACE:
					# print('space')
					playerPlane.keyHandle('space')
					bullet_sound.play()


if __name__ == '__main__':
	human_play()
