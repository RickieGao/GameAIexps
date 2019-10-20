# -*- coding: utf-8 -*-

import pygame

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 800


# 子弹类
class Bullet(object):
	def __init__(self, screen, plane):
		# 子弹图片
		bulletImageName = 'resources/image/bullet1.png'
		# 导入子弹图片
		self.image = pygame.image.load(bulletImageName)

		# 图片大小
		self.image_width = 9
		self.image_height = 21

		# 图片位置
		self.x = plane.x + plane.image_width // 2 - self.image_width // 2
		self.y = plane.y + self.image_height 

		# 设置速度
		self.speed = 12

		self.window = screen

	def getPos(self):
		return [self.x, self.y, self.image_width, self.image_height]

	def move(self):
		self.y -= self.speed

	def draw(self):
		self.window.blit(self.image, (self.x, self.y))


# 玩家飞机类
class Player(object):
	def __init__(self, screen):
		# 储存子弹列表
		self.bulletList = []

		# 飞机图片
		planeImageName = 'resources/image/hero1.png'
		# 导入飞机图片
		self.image = pygame.image.load(planeImageName)

		# 图片大小
		self.image_width = 100
		self.image_height = 124

		# 设置默认坐标
		self.x = 180
		self.y = 600

		# 设置速度
		self.speed = 20

		self.window =  screen

		# 设置飞机名称
		self.name = 'player'

	def getPos(self):
		return [self.x, self.y, self.image_width, self.image_height]

	def crash(self):
		# 图片
		crashImageName = 'resources/image/hero_blowup_n3.png'
		# 导入图片
		self.image = pygame.image.load(crashImageName)

	# 发射子弹，创建一个新的子弹对象
	def shoot(self):
		bullet = Bullet(self.window, self)
		self.bulletList.append(bullet)

	# 清楚不使用的子弹列表元素
	def bulletClear(self):
		num = len(self.bulletList)
		for i in range(0,num):
			if self.bulletList[i].y <= 0:
				del  self.bulletList[i]
				break

	# 飞机上移
	def moveUp(self):
		if self.y <= 0:
			self.y = 0
		else:
			self.y -= self.speed

	# 飞机下移
	def moveDown(self):
		if self.y >= SCREEN_HEIGHT - self.image_height:
			self.y = SCREEN_HEIGHT - self.image_height
		else:
			self.y += self.speed

	# 飞机左移
	def moveLeft(self):
		if self.x <= 0:
			self.x = 0
		else:
			self.x -= self.speed

	# 飞机右移
	def moveRight(self):
		if self.x >= SCREEN_WIDTH - self.image_width:
			self.x = SCREEN_WIDTH - self.image_width
		else:
			self.x += self.speed

	# 绘制图形
	def draw(self):
		self.window.blit(self.image, (self.x, self.y))

	# 按键操作处理
	def keyHandle(self, keyValue):
		if keyValue == 'left':
			self.moveLeft()
		elif keyValue == 'right':
			self.moveRight()
		elif keyValue == 'up':
			self.moveUp()
		elif keyValue == 'down':
			self.moveDown()
		elif keyValue == 'space':
			# bullet = Bullet(self.x, self.y)
			self.shoot()


class Enemy(object):
	def __init__(self, screen, x=0, y=0):
		# 图片
		enemyImageName = 'resources/image/enemy1.png'
		# 导入图片
		self.image = pygame.image.load(enemyImageName)

		# 图片大小
		self.image_width = 51
		self.image_height = 39

		self.speed = 5

		# 图片位置
		self.x = x
		self.y = y

		self.window = screen

		self.direction = 'right'

		self.name = 'enemy'

	def getPos(self):
		return [self.x, self.y, self.image_width, self.image_height]

	# 检测是否出界
	def checkOut(self):
		if self.y >= SCREEN_HEIGHT - self.image_height:
			return True
		else:
			return False

	def move(self):
		self.y += self.speed

	def crash(self):
		# 图片
		crashImageName = 'resources/image/enemy1_down3.png'
		# 导入图片
		self.image = pygame.image.load(crashImageName)

	def draw(self):
		self.window.blit(self.image, (self.x, self.y))
