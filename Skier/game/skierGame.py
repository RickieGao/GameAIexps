import pygame, sys, random

skier_images = ["./bg_img/skier_down.png", "./bg_img/skier_right1.png", "./bg_img/skier_right2.png",
                 "./bg_img/skier_left2.png", "./bg_img/skier_left1.png"]


class SkierClass(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("./bg_img/skier_down.png")
        self.rect = self.image.get_rect()
        self.rect.center = [320, 100]
        self.angle = 0

    def turn(self, direction):
        self.angle = self.angle + direction
        if self.angle < -2:
            self.angle = -2
        if self.angle > 2:
            self.angle = 2
        center = self.rect.center
        self.image = pygame.image.load(skier_images[self.angle])
        self.rect = self.image.get_rect()
        self.rect.center = center
        speed = [self.angle, 6 - abs(self.angle) * 2]
        return speed

    def move(self, speed):
        self.rect.centerx = self.rect.centerx + speed[0]
        if self.rect.centerx < 20:
            self.rect.centerx = 20
        if self.rect.centerx > 620:
            self.rect.centerx = 620


class ObstacleClass(pygame.sprite.Sprite):
    def __init__(self, image_file, location, type):
        pygame.sprite.Sprite.__init__(self)
        self.image_file = image_file
        self.image = pygame.image.load(image_file)
        self.location = location
        self.rect = self.image.get_rect()
        self.rect.center = location
        self.type = type
        self.passed = False

    def scroll(self, terrainPos):
        self.rect.centery = self.location[1] - terrainPos


def create_map(start, end):
    obstacles = pygame.sprite.Group()
    locations = []
    gates = pygame.sprite.Group()
    for i in range(10):
        row = random.randint(start, end)
        col = random.randint(0, 9)
        location = [col * 64 + 20, row * 64 + 20]
        if not (location in locations):
            locations.append(location)
            type = random.choice(["tree", "flag"])
            if type == "tree":
                img = "./bg_img/skier_tree.png"
            elif type == "flag":
                img = "./bg_img/skier_flag.png"
            obstacle = ObstacleClass(img, location, type)
            obstacles.add(obstacle)
    return obstacles


def animate(screen, skier, obstacles):
    screen.fill([255, 255, 255])
    pygame.display.update(obstacles.draw(screen))
    screen.blit(skier.image, skier.rect)
    # screen.blit(score_text, [10, 10])
    pygame.display.flip()


def updateObstacleGroup(map0, map1):
    obstacles = pygame.sprite.Group()
    for ob in map0:
        obstacles.add(ob)
    for ob in map1:
        obstacles.add(ob)
    return obstacles


class GameState:

    def __init__(self):
        self.screen = pygame.display.set_mode([640, 640])
        # self.clock = pygame.time.Clock()
        self.skier = SkierClass()
        self.speed = [0, 6]
        self.map_position = 0
        # self.points = 0
        self.map0 = create_map(20, 29)
        self.map1 = create_map(10, 19)
        self.activeMap = 0

        self.obstacles = updateObstacleGroup(self.map0, self.map1)

        # self.font = pygame.font.Font(None, 50)

    def frame_step(self, input_actions):
        # self.clock.tick(30)

        isterminate = False
        reward = 0.1

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT: sys.exit()
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             speed = skier.turn(-1)
        #         elif event.key == pygame.K_RIGHT:
        #             speed = skier.turn(1)

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')
        if input_actions[0] == 1:
            pass
        if input_actions[1] == 1:
            self.speed = self.skier.turn(-1)
        if input_actions[2] == 1:
            self.speed = self.skier.turn(1)

        self.skier.move(self.speed)
        self.map_position += self.speed[1]

        if self.map_position >= 640 and self.activeMap == 0:
            self.activeMap = 1
            self.map0 = create_map(20, 29)
            self.obstacles = updateObstacleGroup(self.map0, self.map1)
        if self.map_position >= 1280 and self.activeMap == 1:
            self.activeMap = 0
            for ob in self.map0:
                ob.location[1] = ob.location[1] - 1280
            self.map_position = self.map_position - 1280
            self.map1 = create_map(10, 19)
            self.obstacles = updateObstacleGroup(self.map0, self.map1)

        for obstacle in self.obstacles:
            obstacle.scroll(self.map_position)

        hit = pygame.sprite.spritecollide(self.skier, self.obstacles, False)
        if hit:
            if hit[0].type == "tree" and not hit[0].passed:
                # self.points -= 100
                reward = -1
                self.skier.image = pygame.image.load("./bg_img/skier_crash.png")
                animate(self.screen, self.skier, self.obstacles)
                # pygame.time.delay(1000)
                pygame.time.delay(10)
                self.skier.image = pygame.image.load("./bg_img/skier_down.png")
                self.skier.angle = 0
                self.speed = [0, 6]
                hit[0].passed = True
                isterminate = True
                self.__init__()
            elif hit[0].type == "flag" and not hit[0].passed:
                # self.points += 10
                reward = 1
                self.obstacles.remove(hit[0])

        # score_text = font.render("Score: " + str(self.points), 1, (0, 0, 0))
        animate(self.screen, self.skier, self.obstacles)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, reward, isterminate
