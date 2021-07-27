import pygame
import sys
import os

current_dir = os.path.dirname(__file__)


def load():
    # path of player with different states
    PLAYER_PATH = (
        current_dir + '/../assets/sprites/redbird-upflap.png',
        current_dir + '/../assets/sprites/redbird-midflap.png',
        current_dir + '/../assets/sprites/redbird-downflap.png'
    )

    # 背景路径
    BACKGROUND_PATH = current_dir + '/../assets/sprites/background-black.png'

    # 游戏中绿色管道路径
    PIPE_PATH = current_dir + '/../assets/sprites/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}  # 图像，声音，撞击的文件

    # numbers sprites for score display
    IMAGES['numbers'] = (
        #  convert_alpha相对于convert，保留了图像的Alpha 通道信息，可以认为是保留了透明的部分，实现了透明转换
        pygame.image.load(current_dir + '/../assets/sprites/0.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/1.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/2.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/3.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/4.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/5.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/6.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/7.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/8.png').convert_alpha(),
        pygame.image.load(current_dir + '/../assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load(current_dir + '/../assets/sprites/base.png').convert_alpha()

    # 判断当前系统平台 来设置声音文件后缀
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    try:
        # sound = pygame.mixer.Sound('_dir')使用指定文件名载入一个音频文件，并创建一个Sound对象。
        SOUNDS['die'] = pygame.mixer.Sound(current_dir + '/../assets/audio/die' + soundExt)  # 死亡
        SOUNDS['hit'] = pygame.mixer.Sound(current_dir + '/../assets/audio/hit' + soundExt)  # 击打
        SOUNDS['point'] = pygame.mixer.Sound(current_dir + '/../assets/audio/point' + soundExt)  # 得分
        SOUNDS['swoosh'] = pygame.mixer.Sound(current_dir + '/../assets/audio/swoosh' + soundExt)  #
        SOUNDS['wing'] = pygame.mixer.Sound(current_dir + '/../assets/audio/wing' + soundExt)  # 翅膀
    except:
        print('音效系统初始化失败！')

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # 加载随机管道样式
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )  # 一个上面的管道 一个下面的管道

    # 得到管道的边界mask
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # 得到player的边界mask
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS


def getHitmask(image):
    # 得到撞击mask
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask
