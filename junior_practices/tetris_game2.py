"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 11/08/2025, 20:36
@Desc    : 

"""

import pygame
import random

# 初始化
pygame.init()

# 背景音乐
pygame.mixer.music.load(r"music/background.mp3")
pygame.mixer.music.set_volume(0.5)  # 范围0.0~1.0
pygame.mixer.music.play(-1)  # -1 表示循环播放
# 消除音效
clear_sound = pygame.mixer.Sound(r"music/clear.wav")
clear_sound.set_volume(0.5)
# 旋转音效
rotate_sound = pygame.mixer.Sound(r"music/rotate.wav")
rotate_sound.set_volume(0.5)

highest_score = 0
WIDTH, HEIGHT = 420, 600
BLOCK_SIZE = 30
COLS, ROWS = 10, 20
GRID_WIDTH = COLS * BLOCK_SIZE
GRID_HEIGHT = ROWS * BLOCK_SIZE
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tetris-Dr. Zhen Chen's game")
PREVIEW_BLOCK_SIZE = 20

# 颜色
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
# COLORS = [
#     (72, 209, 204),  # 亮青色，柔和的蓝绿
#     (65, 105, 225),  # 皇家蓝
#     (255, 140, 0),  # 深橙色
#     (255, 215, 0),  # 金黄色
#     (60, 179, 113),  # 中海蓝绿
#     (220, 20, 60),  # 猩红
#     (147, 112, 219)  # 中等兰花紫
# ]
COLORS = [
    (64, 224, 208),   # 深浅适中的亮青色 (Turquoise)
    (70, 130, 180),   # 钢蓝 (Steel Blue)
    (255, 165, 0),    # 橙色 (Orange)
    (255, 223, 0),    # 鲜黄 (Gold)
    (46, 139, 87),    # 海洋绿 (Sea Green)
    (255, 0, 0),      # 纯红 (Red)
    (138, 43, 226)    # 蓝紫色 (Blue Violet)
]

# 形状
SHAPES = [
    # 每个小列表是“行”，
    # 里面的数字代表“列”，
    # 1 是方块实际存在的位置，0 是空白格
    [[1, 1, 1, 1]],  # |
    [[1, 0, 0], [1, 1, 1]],  # |___ (L)
    [[0, 0, 1], [1, 1, 1]],  # ___|
    [[1, 1], [1, 1]],  # square
    [[0, 1, 1], [1, 1, 0]],  # __--
    [[1, 1, 0], [0, 1, 1]],  # --__ (z)
    [[0, 1, 0], [1, 1, 1]]  # T
]


class Piece:
    def __init__(self, x, y, shape):
        self.x, self.y = x, y
        self.y_float = float(y)
        self.shape = shape
        self.color = random.choice(COLORS)

    def rotate(self):
        """
            顺时针旋转90度
        """
        self.shape = list(zip(*self.shape[::-1]))
        rotate_sound.play()


def load_highscore(filename="highscore.txt"):
    try:
        global highest_score
        with open(filename, "r") as f:
            highest_score = int(f.read())
    except (FileNotFoundError, ValueError):
        highest_score = 0  # 文件不存在或内容不正确，默认0分
    return highest_score


def save_highscore(score, filename="highscore.txt"):
    with open(filename, "w") as f:
        f.write(str(score))


# grid 存贮了每一个网格的颜色
def create_grid():
    return [[BLACK for _ in range(COLS)] for _ in range(ROWS)]


def valid_position(piece, grid):
    """
       判断方块是否处于有效位置
    Args:
        piece:
        grid:

    Returns:

    """
    # shape 是一个二维列表
    for i, row in enumerate(piece.shape):
        for j, cell in enumerate(row):
            if cell:
                nx, ny = piece.x + j, int(piece.y) + i
                # 最后一个条件表示碰到了其他的方块
                if nx < 0 or nx >= COLS or ny >= ROWS or (ny >= 0 and grid[ny][nx] != BLACK):
                    return False
    return True


def lock_piece(piece, grid):
    """
        将下落不动的方块染色固定
    Args:
        piece:
        grid:
    """
    for i, row in enumerate(piece.shape):
        for j, cell in enumerate(row):
            if cell:
                grid[int(piece.y) + i][piece.x + j] = piece.color


def blink_lines(surface, grid, lines_to_clear, blink_times=4, blink_delay=100):
    for _ in range(blink_times): # 闪烁次数
        # 用白色(或其它颜色)覆盖消除行
        for row in lines_to_clear:
            for col in range(COLS):
                rect = pygame.Rect(col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(surface, (255, 255, 255), rect)  # 白色闪烁

        pygame.display.update()
        pygame.time.wait(blink_delay)

        # 恢复正常颜色显示
        draw_grid(SCREEN, grid)
        pygame.display.update()
        pygame.time.wait(blink_delay)

def clear_lines(grid):
    cleared = 0
    i = ROWS - 1  # 从最后一行开始检查
    lines_to_clear = []

    # 先找到要消除的行
    for i in range(ROWS):
        if all(cell != BLACK for cell in grid[i]):
            lines_to_clear.append(i)

    if lines_to_clear:
        # 闪烁动画
        blink_lines(SCREEN, grid, lines_to_clear)

    # 然后消除行
    i = ROWS - 1
    while i >= 0:
        if all(cell != BLACK for cell in grid[i]):
            del grid[i]
            grid.insert(0, [BLACK] * COLS)
            cleared += 1
            clear_sound.play()
            # 不递减i
        else:
            i -= 1
    return cleared


def draw_block_with_shadow(surface, color, rect):
    # 画底色（暗色版）
    dark_color = tuple(max(c - 40, 0) for c in color)
    pygame.draw.rect(surface, dark_color, rect)

    # 画上层（偏亮色块，位置微调制造高光）
    highlight_rect = rect.inflate(-6, -6)
    light_color = tuple(min(c + 40, 255) for c in color)
    pygame.draw.rect(surface, light_color, highlight_rect)

    # 画边框
    pygame.draw.rect(surface, GRAY, rect, 1)


def draw_blocks(surface, grid):
    for y in range(ROWS):
        for x in range(COLS):
            color = grid[y][x]
            if color != BLACK:
                rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                draw_block_with_shadow(surface, color, rect)  # 填充颜色
                pygame.draw.rect(surface, GRAY, rect, 1)  # 画边框线，线宽1

def draw_grid(surface, grid):
    for row in range(ROWS):
        for col in range(COLS):
            color = grid[row][col]
            rect = pygame.Rect(col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (40, 40, 40), rect, 1)

def draw_grid_lines(surface):
    # 这里你可以选择是否保留背景网格线，如果想要更简洁可以注释掉
    for x in range(COLS):
        pygame.draw.line(surface, GRAY, (x * BLOCK_SIZE, 0), (x * BLOCK_SIZE, GRID_HEIGHT))
    for y in range(ROWS):
        pygame.draw.line(surface, GRAY, (0, y * BLOCK_SIZE), (GRID_WIDTH, y * BLOCK_SIZE))


def draw_text(surface, text, size, x, y, color=WHITE):
    font = pygame.font.SysFont("Arial", size, bold=True)
    label = font.render(text, True, color)
    surface.blit(label, (x, y))


def draw_piece(surface, piece):
    for i, row in enumerate(piece.shape):
        for j, cell in enumerate(row):
            if cell:
                px = (piece.x + j) * BLOCK_SIZE
                py = (int(piece.y_float) + i) * BLOCK_SIZE
                rect = pygame.Rect(px, py, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(surface, piece.color, rect)  # 填充颜色
                pygame.draw.rect(surface, GRAY, rect, 1)  # 画边框线，线宽1


def main():
    global highest_score
    load_highscore(filename="highscore.txt")
    grid = create_grid()  # grid 是一个2维数组
    clock = pygame.time.Clock()  # 控制游戏运行帧率FPS,即每秒刷新次数

    # COLS // 2 - 2 是初始方块横向的位置，
    # 让方块大致出现在网格中间偏左一点（因为方块宽度最大4格，减2是为了居中）
    # y=0 表示最上面的位置
    current_piece = Piece(COLS // 2 - 2, 0, random.choice(SHAPES))
    next_piece = Piece(COLS // 2 - 2, 0, random.choice(SHAPES))

    fall_speed = 1.0  # 每格下落时间（秒）
    score, level, lines_cleared = 0, 1, 0
    running = True

    down_pressed_time = None
    down_threshold = 200  # 长按毫秒结点

    while running:
        # FPS 60 帧
        dt = clock.tick(60) / 1000  # 秒

        # 平滑下落时，限制下落速度不跳过检测
        # dt / fall_speed 表示这帧下落多少格
        # 防止下落速度过快
        current_piece.y_float += min(dt / fall_speed, 0.5)

        # 一格一格检测碰撞
        if current_piece.y_float >= current_piece.y + 1:
            current_piece.y += 1
            if not valid_position(current_piece, grid):
                current_piece.y -= 1
                current_piece.y_float = float(current_piece.y)
                lock_piece(current_piece, grid)
                cleared = clear_lines(grid)
                if cleared > 0:
                    lines_cleared += cleared
                    score += {1: 100, 2: 300, 3: 500, 4: 800}[cleared] * level
                    if lines_cleared // 10 > level - 1:  # 每清除 10 行升一级
                        level += 1
                        fall_speed = max(0.1, fall_speed - 0.05)
                current_piece = next_piece
                current_piece.x, current_piece.y, current_piece.y_float = COLS // 2 - 2, 0, 0.0
                next_piece = Piece(COLS // 2 - 2, 0, random.choice(SHAPES))
                if not valid_position(current_piece, grid):
                    draw_text(SCREEN, "GAME OVER", 40, 60, 250, (255, 0, 0))
                    pygame.display.update()
                    pygame.time.wait(2000)
                    running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # QUIT 表示窗口关闭
                running = False

            elif event.type == pygame.KEYDOWN:  # 表示键盘某个键被按下
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not valid_position(current_piece, grid):
                        current_piece.x += 1
                elif event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not valid_position(current_piece, grid):
                        current_piece.x -= 1
                elif event.key == pygame.K_DOWN:
                    down_pressed_time = pygame.time.get_ticks()
                    current_piece.y_float += 0.1  # 软降一次
                elif event.key == pygame.K_UP:
                    current_piece.rotate()
                    # 如果旋转后位置不合法，旋转3次（等于逆时针旋转一次，回到原始状态）
                    if not valid_position(current_piece, grid):
                        for _ in range(3):
                            current_piece.rotate()

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN and down_pressed_time is not None:
                    press_duration = pygame.time.get_ticks() - down_pressed_time
                    if press_duration >= down_threshold:
                        # 硬降到底
                        while valid_position(current_piece, grid):
                            current_piece.y += 1
                        # 因为循环退出时方块位置已经超出合法范围（多移动了一格），
                        # 所以退回一格，恢复到最后一个合法的位置
                        current_piece.y -= 1
                        current_piece.y_float = float(current_piece.y)
                    down_pressed_time = None

        SCREEN.fill(BLACK)
        draw_blocks(SCREEN, grid)
        draw_piece(SCREEN, current_piece)
        draw_grid_lines(SCREEN)

        # 预览下一个方块
        draw_text(SCREEN, "Next:", 30, GRID_WIDTH + 30, 20)
        preview = Piece(0, 0, next_piece.shape)
        preview.color = next_piece.color

        preview_x = GRID_WIDTH + 10
        preview_y = 50

        shape_rows = len(preview.shape)
        shape_cols = len(preview.shape[0])

        # centered the preview
        offset_x = (4 - shape_cols) * PREVIEW_BLOCK_SIZE // 2
        offset_y = (4 - shape_rows) * PREVIEW_BLOCK_SIZE // 2

        for i, row in enumerate(preview.shape):
            for j, cell in enumerate(row):
                if cell:
                    px = preview_x + offset_x + j * PREVIEW_BLOCK_SIZE
                    py = preview_y + offset_y + i * PREVIEW_BLOCK_SIZE
                    rect = pygame.Rect(px, py, PREVIEW_BLOCK_SIZE, PREVIEW_BLOCK_SIZE)
                    pygame.draw.rect(SCREEN, preview.color, rect)
                    pygame.draw.rect(SCREEN, GRAY, rect, 1)

        # 分数、等级、行数
        if score > highest_score:
            highest_score = score
        draw_text(SCREEN, f"Highest score:", 15, GRID_WIDTH + 10, 150)
        draw_text(SCREEN, f"{highest_score:^30}", 15, GRID_WIDTH + 10, 180)
        draw_text(SCREEN, f"Score: {score:<30}", 15, GRID_WIDTH + 10, 210)
        draw_text(SCREEN, f"Level: {level}", 15, GRID_WIDTH + 10, 240)
        draw_text(SCREEN, f"Lines: {lines_cleared}", 15, GRID_WIDTH + 10, 270)

        draw_text(SCREEN, f"↑:  rotate", 15, GRID_WIDTH + 10, 350)
        draw_text(SCREEN, f"↓:  drop/fast drop", 15, GRID_WIDTH + 10, 380)
        draw_text(SCREEN, f"←:  left", 15, GRID_WIDTH + 10, 410)
        draw_text(SCREEN, f"→:  right", 15, GRID_WIDTH + 10, 440)

        pygame.display.update()

    save_highscore(highest_score)
    pygame.quit()


if __name__ == "__main__":
    main()