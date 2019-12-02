import pygame
from pygame.locals import *
import sys, random, time, math


class GameWindow(object):
    # 创建游戏窗口类

    def __init__(self, *args, **kw):
        # 设置窗口尺寸
        self.window_length = 600
        self.window_wide = 500
        # 绘制游戏窗口
        self.game_window = pygame.display.set_mode((self.window_length, self.window_wide))
        # 设置游戏窗口标题
        pygame.display.set_caption("breakout")
        # 定义游戏窗口背景颜色参数
        self.window_color = (255, 255, 255)
        # 绘制游戏窗口背景颜色

    def backgroud(self):
        self.game_window.fill(self.window_color)


class Ball(object):
    # 创建球类

    def __init__(self, *args, **kw):
        # 设置球的半径、颜色、移动速度参数
        self.ball_color = (0, 0, 0)
        self.move_x = 9
        self.move_y = 9
        self.radius = 10

        # 设置球的初始位置
        # self.ball_x = self.window_length // 2
        self.ball_x = self.bat_mid
        self.ball_y = self.window_wide - self.bat_wide - self.radius
        # 绘制球
        pygame.draw.circle(self.game_window, self.ball_color, (self.ball_x, self.ball_y), int(self.radius))

    def ballready(self):
        # 设置球的初始位置
        # self.ball_x = self.window_length // 2
        self.ball_x = self.bat_mid
        self.ball_y = self.window_wide - self.bat_wide - self.radius
        # 绘制球，设置反弹触发条件
        pygame.draw.circle(self.game_window, self.ball_color, (self.ball_x, self.ball_y), self.radius)

    def ball_move(self):
        # 绘制球，设置反弹触发条件
        pygame.draw.circle(self.game_window, self.ball_color, (self.ball_x, self.ball_y), self.radius)
        self.ball_x += self.move_x
        self.ball_y -= self.move_y
        # 调用碰撞检测函数
        self.ball_window()
        self.ball_bat()
        # 设置游戏失败条件
        if self.ball_y > 520:
            self.over_sign = 1
            self.reward = -5


class Bat(object):
    # 创建球拍类

    def __init__(self, *args, **kw):

        # 设置球拍参数
        self.bat_color = (0, 0, 0)
        self.bat_length = 50
        self.bat_wide = 10
        # 球拍单步运动步长
        self.bat_path_length = 10
        # self.bat_x = self.window_length // 2 - self.bat_length // 2
        self.bat_x = random.randint(0, self.window_length - self.bat_length)
        self.bat_y = self.window_wide - self.bat_wide
        self.bat_mid = self.bat_x + self.bat_length // 2

        # 画出球拍
        pygame.draw.rect(self.game_window, self.bat_color, (self.bat_x, self.bat_y, self.bat_length, self.bat_wide))

    def bat_move(self, input_actions):

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: move to the left
        # input_actions[1] == 1: do nothing
        # input_actions[2] == 1: move to the right

        if input_actions[0] == 1:
            if (self.bat_x - self.bat_path_length) > 0:
                self.bat_x = self.bat_x - self.bat_path_length
            else:
                self.bat_x = 0
        if input_actions[2] == 1:
            count = self.bat_x + self.bat_length + self.bat_path_length
            if count <= self.window_length:
                self.bat_x = self.bat_x + self.bat_path_length
            else:
                self.bat_x = self.window_length - self.bat_length
        pygame.draw.rect(self.game_window, self.bat_color, (self.bat_x, self.bat_y, self.bat_length, self.bat_wide))


class Brick(object):
    # 创建砖块类
    def __init__(self, *args, **kw):
        # 设置砖块颜色参数
        self.brick_color = (0, 0, 0)
        self.brick_list = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1]]
        self.brick_length = 100
        self.brick_wide = 20
        self.brick_x = 0
        self.brick_y = 0

    def brick_arrange(self):
        for i in range(5):
            for j in range(6):
                self.brick_x = j * self.brick_length
                self.brick_y = i * self.brick_wide + 40
                if self.brick_list[i][j] == 1:
                    # 绘制砖块
                    pygame.draw.rect(self.game_window, self.brick_color,
                                     (self.brick_x, self.brick_y, self.brick_length, self.brick_wide))
                    # 调用碰撞检测函数
                    self.ball_brick()
                    if self.distanceb < self.radius:
                        self.brick_list[i][j] = 0
                        self.score += self.point
                        self.reward = 1

        # 设置游戏胜利条件
        if self.brick_list == [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]]:
            self.win_sign = 1
            self.reward += 10


class Score(object):
    # 创建分数类

    def __init__(self, *args, **kw):
        # 设置初始分数
        self.score = 0
        # 设置分数字体
        self.score_font = pygame.font.SysFont('arial', 20)
        # 设置初始加分点数
        self.point = 1
        # 设置初始接球次数
        self.frequency = 0

    def count_score(self):
        # 绘制玩家分数
        my_score = self.score_font.render(str(self.score), False, (0, 0, 0))
        self.game_window.blit(my_score, (555, 15))


class Collision(object):
    # 碰撞检测类

    # 球与窗口边框的碰撞检测
    def ball_window(self):
        if self.ball_x <= self.radius or self.ball_x >= (self.window_length - self.radius):
            self.move_x = -self.move_x
        if self.ball_y <= self.radius:
            self.move_y = -self.move_y

    # 球与球拍的碰撞检测
    def ball_bat(self):
        # 定义碰撞标识
        self.collision_sign_x = 0
        self.collision_sign_y = 0

        if self.ball_x < self.bat_x:
            self.closestpoint_x = self.bat_x
            self.collision_sign_x = 1
        elif self.ball_x > self.bat_x + self.bat_length:
            self.closestpoint_x = self.bat_x + self.bat_length
            self.collision_sign_x = 2
        else:
            self.closestpoint_x = self.ball_x
            self.collision_sign_x = 3

        if self.ball_y < (self.window_wide - self.bat_wide):
            self.closestpoint_y = (self.window_wide - self.bat_wide)
            self.collision_sign_y = 1
        elif self.ball_y > self.window_wide:
            self.closestpoint_y = self.window_wide
            self.collision_sign_y = 2
        else:
            self.closestpoint_y = self.ball_y
            self.collision_sign_y = 3
        # 定义球拍到圆心最近点与圆心的距离
        self.distance = math.sqrt(
            math.pow(self.closestpoint_x - self.ball_x, 2) + math.pow(self.closestpoint_y - self.ball_y, 2))
        # 球在球拍上左、上中、上右3种情况的碰撞检测
        if self.distance < self.radius and self.collision_sign_y == 1 and (
                self.collision_sign_x == 1 or self.collision_sign_x == 2):
            if self.collision_sign_x == 1 and self.move_x > 0:
                self.move_x = - self.move_x
                self.move_y = - self.move_y
                self.reward += 1
            if self.collision_sign_x == 1 and self.move_x < 0:
                self.move_y = - self.move_y
                self.reward += 1
            if self.collision_sign_x == 2 and self.move_x < 0:
                self.move_x = - self.move_x
                self.move_y = - self.move_y
                self.reward += 1
            if self.collision_sign_x == 2 and self.move_x > 0:
                self.move_y = - self.move_y
                self.reward += 1
        if self.distance < self.radius and self.collision_sign_y == 1 and self.collision_sign_x == 3:
            self.move_y = - self.move_y
            self.reward += 1
        # 球在球拍左、右两侧中间的碰撞检测
        if self.distance < self.radius and self.collision_sign_y == 3:
            self.move_x = - self.move_x

    # 球与砖块的碰撞检测
    def ball_brick(self):
        # 定义碰撞标识
        self.collision_sign_bx = 0
        self.collision_sign_by = 0

        if self.ball_x < self.brick_x:
            self.closestpoint_bx = self.brick_x
            self.collision_sign_bx = 1
        elif self.ball_x > self.brick_x + self.brick_length:
            self.closestpoint_bx = self.brick_x + self.brick_length
            self.collision_sign_bx = 2
        else:
            self.closestpoint_bx = self.ball_x
            self.collision_sign_bx = 3

        if self.ball_y < self.brick_y:
            self.closestpoint_by = self.brick_y
            self.collision_sign_by = 1
        elif self.ball_y > self.brick_y + self.brick_wide:
            self.closestpoint_by = self.brick_y + self.brick_wide
            self.collision_sign_by = 2
        else:
            self.closestpoint_by = self.ball_y
            self.collision_sign_by = 3
        # 定义砖块到圆心最近点与圆心的距离
        self.distanceb = math.sqrt(
            math.pow(self.closestpoint_bx - self.ball_x, 2) + math.pow(self.closestpoint_by - self.ball_y, 2))
        # 球在砖块上左、上中、上右3种情况的碰撞检测
        if self.distanceb < self.radius and self.collision_sign_by == 1 and (
                self.collision_sign_bx == 1 or self.collision_sign_bx == 2):
            if self.collision_sign_bx == 1 and self.move_x > 0:
                self.move_x = - self.move_x
                self.move_y = - self.move_y
            if self.collision_sign_bx == 1 and self.move_x < 0:
                self.move_y = - self.move_y
            if self.collision_sign_bx == 2 and self.move_x < 0:
                self.move_x = - self.move_x
                self.move_y = - self.move_y
            if self.collision_sign_bx == 2 and self.move_x > 0:
                self.move_y = - self.move_y
        if self.distanceb < self.radius and self.collision_sign_by == 1 and self.collision_sign_bx == 3:
            self.move_y = - self.move_y
        # 球在砖块下左、下中、下右3种情况的碰撞检测
        if self.distanceb < self.radius and self.collision_sign_by == 2 and (
                self.collision_sign_bx == 1 or self.collision_sign_bx == 2):
            if self.collision_sign_bx == 1 and self.move_x > 0:
                self.move_x = - self.move_x
                self.move_y = - self.move_y
            if self.collision_sign_bx == 1 and self.move_x < 0:
                self.move_y = - self.move_y
            if self.collision_sign_bx == 2 and self.move_x < 0:
                self.move_x = - self.move_x
                self.move_y = - self.move_y
            if self.collision_sign_bx == 2 and self.move_x > 0:
                self.move_y = - self.move_y
        if self.distanceb < self.radius and self.collision_sign_by == 2 and self.collision_sign_bx == 3:
            self.move_y = - self.move_y
        # 球在砖块左、右两侧中间的碰撞检测
        if self.distanceb < self.radius and self.collision_sign_by == 3:
            self.move_x = - self.move_x


class Main(GameWindow, Bat, Ball, Brick, Collision, Score):
    # 创建主程序类

    def __init__(self, *args, **kw):
        super(Main, self).__init__(*args, **kw)
        super(GameWindow, self).__init__(*args, **kw)
        super(Ball, self).__init__(*args, **kw)
        super(Brick, self).__init__(*args, **kw)
        super(Collision, self).__init__(*args, **kw)
        super(Score, self).__init__(*args, **kw)
        super(Bat, self).__init__(*args)
        # 游戏标识
        self.win_sign = 0
        self.over_sign = 0
        self.start_sign = 1
        self.ballready()

    def frame_step(self, input_actions):
        pygame.event.pump()

        if self.start_sign == 0:
            self.__init__()

        self.reward = 0.1
        terminal = False

        self.backgroud()
        self.bat_move(input_actions)
        self.count_score()

        self.ball_move()

        self.brick_arrange()

        if self.over_sign == 1 or self.win_sign == 1:
            terminal = True
            self.__init__()

        # 更新游戏窗口
        pygame.display.update()
        # 控制游戏窗口刷新频率
        # time.sleep(0.010)
        # clock = pygame.time.Clock()
        # clock.tick(60)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.bat_mid = self.bat_x + self.bat_length // 2
        # print(self.ball_x, self.bat_mid)
        return image_data, self.reward, terminal, self.ball_x, self.bat_mid


#
# def human_play():
#     running = 1
#     while running:
#         for event in pygame.event.get():
#             if event.type == QUIT:
#                 pygame.quit()
#                 sys.exit()
#             if event.type == KEYDOWN:
#                 if event.key == K_LEFT:
#                     reward, terminal, image_date, ball, bat = Main.frame_step([1, 0, 0])
#                 if event.key == K_RIGHT:
#                     reward, terminal, image_date, ball, bat = Main.frame_step([0, 0, 1])
#             # print(reward)
#             # if terminal == 1:
#             #     pygame.quit()
#             #     sys.exit()
#
#
# if __name__ == '__main__':
#     pygame.init()
#     pygame.font.init()
#     human_play()
























