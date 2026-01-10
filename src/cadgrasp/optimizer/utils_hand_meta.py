from enum import Enum, IntEnum
import numpy as np
import sys

# 定义指代手指的枚举
class Finger(IntEnum):
    PALM    = 0     # 手掌
    THUMB   = 1     # 拇指
    INDEX   = 2     # 食指
    MIDDLE  = 3     # 中指
    RING    = 4     # 无名指
    PINKY   = 5     # 小指

# 定义掌侧和背侧的枚举
class IsPalmar(IntEnum):
    DORSAL  = 0     # 背侧
    PALMAR  = 1     # 掌侧

# 定义关节位置的枚举
class Joint(IntEnum):
    PALM    = 0     # 手掌
    BASE    = 1     # 基关节
    MIDDLE  = 2     # 中间关节
    DISTAL  = 3     # 远端关节
    
def get_finger_and_joint(robot_name, link_name):
    ret = [None, None]
    if robot_name == 'shadow':
        if link_name in ['wrist', 'palm', 'thbase', 'lfmetacarpal']:
            ret[0] = Finger.PALM
            ret[1] = Joint.PALM
        elif link_name in ['thproximal', 'thmiddle', 'thdistal', 'thhub', 'thtip']:
            ret[0] = Finger.THUMB
        elif link_name in ['ffproximal', 'ffmiddle', 'ffdistal', 'ffknuckle', 'fftip']:
            ret[0] = Finger.INDEX
        elif link_name in ['mfproximal', 'mfmiddle', 'mfdistal', 'mfknuckle', 'mftip']:
            ret[0] = Finger.MIDDLE
        elif link_name in ['rfproximal', 'rfmiddle', 'rfdistal', 'rfknuckle', 'rftip']:
            ret[0] = Finger.RING
        elif link_name in ['lfproximal', 'lfmiddle', 'lfdistal', 'lfknuckle', 'lftip']:
            ret[0] = Finger.PINKY

        if link_name in ['thproximal', 'ffproximal', 'mfproximal', 'rfproximal', 'lfproximal', 'thhub', 'ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle']:
            ret[1] = Joint.BASE
        elif link_name in ['thmiddle', 'ffmiddle', 'mfmiddle', 'rfmiddle', 'lfmiddle']:
            ret[1] = Joint.MIDDLE
        elif link_name in ['thdistal', 'ffdistal', 'mfdistal', 'rfdistal', 'lfdistal', 'thtip', 'fftip', 'mftip', 'rftip', 'lftip']:
            ret[1] = Joint.DISTAL

    return ret


class InfoData:
    def __init__(self, finger=Finger.THUMB, is_palmar=IsPalmar.DORSAL, joint=Joint.BASE):
        # 检查输入的合法性
        if not isinstance(finger, Finger):
            raise ValueError("Invalid finger type.")
        if not isinstance(is_palmar, IsPalmar):
            raise ValueError("Invalid palmar/dorsal value.")
        if not isinstance(joint, Joint):
            raise ValueError("Invalid joint type.")
        
        # 通过位运算将各个字段组合成一个8比特整数
        self.data = (finger & 0b111) | ((is_palmar & 0b1) << 3) | ((joint & 0b11) << 4)
    
    def from_array(arr):
        arr = arr.astype(int)
        result = np.zeros((arr.shape[0], 3), dtype=int)
        result[:, 0] = arr % 8
        result[:, 1] = (arr >> 3) % 2
        result[:, 2] = (arr >> 4) % 4
        return result

    # 获取 finger 字段
    def get_finger(self):
        return Finger(self.data & 0b111)  # 提取低3位

    # 获取 is_palmar 字段
    def get_is_palmar(self):
        return IsPalmar((self.data >> 3) & 0b1)  # 提取第4位

    # 获取 joint 字段
    def get_joint(self):
        return Joint((self.data >> 4) & 0b11)  # 提取第5-6位

    # 打印结构的二进制表示
    def to_bin(self):
        return f'{self.data:08b}'  # 将整数转换为8位二进制字符串
