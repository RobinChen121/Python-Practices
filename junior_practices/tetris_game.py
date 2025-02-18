"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/18 20:50
Description: 
    

"""
import pygame
import random

# 初始化 Pygame
pygame.init()

# 屏幕大小
WIDTH, HEIGHT = 300, 600
GRID_SIZE = 30
COLUMNS = WIDTH // GRID_SIZE
ROWS = HEIGHT // GRID_SIZE

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
COLORS = [(0, 255, 255), (0, 0, 255), (255, 165, 0),
          (255, 255, 0), (0, 255, 0), (128, 0, 128), (255, 0, 0)]

# 俄罗斯方块的形状
SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]]
]


class Tetrimino:
    def __init__(self):
        self.shape = random.choice(SHAPES)
        self.color = random.choice(COLORS)
        self.x = COLUMNS // 2 - len(self.shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]


def draw_grid(screen):
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))


def draw_tetrimino(screen, tetrimino):
    for y, row in enumerate(tetrimino.shape):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(screen, tetrimino.color,
                                 ((tetrimino.x + x) * GRID_SIZE, (tetrimino.y + y) * GRID_SIZE,
                                  GRID_SIZE, GRID_SIZE))


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    tetrimino = Tetrimino()

    while running:
        screen.fill(BLACK)
        draw_grid(screen)
        draw_tetrimino(screen, tetrimino)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    tetrimino.x -= 1
                elif event.key == pygame.K_RIGHT:
                    tetrimino.x += 1
                elif event.key == pygame.K_DOWN:
                    tetrimino.y += 1
                elif event.key == pygame.K_UP:
                    tetrimino.rotate()

        tetrimino.y += 1  # 自动下落
        pygame.display.flip()
        clock.tick(2)  # 控制游戏速度

    pygame.quit()


if __name__ == "__main__":
    main()
