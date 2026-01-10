from enum import Enum, IntEnum
import numpy as np
import sys

# Enum for finger identification
class Finger(IntEnum):
    PALM    = 0     # Palm
    THUMB   = 1     # Thumb
    INDEX   = 2     # Index finger
    MIDDLE  = 3     # Middle finger
    RING    = 4     # Ring finger
    PINKY   = 5     # Pinky finger

# Enum for palmar/dorsal side
class IsPalmar(IntEnum):
    DORSAL  = 0     # Dorsal side
    PALMAR  = 1     # Palmar side

# Enum for joint position
class Joint(IntEnum):
    PALM    = 0     # Palm
    BASE    = 1     # Base joint
    MIDDLE  = 2     # Middle joint
    DISTAL  = 3     # Distal joint
    
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
        # Validate input types
        if not isinstance(finger, Finger):
            raise ValueError("Invalid finger type.")
        if not isinstance(is_palmar, IsPalmar):
            raise ValueError("Invalid palmar/dorsal value.")
        if not isinstance(joint, Joint):
            raise ValueError("Invalid joint type.")
        
        # Combine fields into an 8-bit integer using bit operations
        self.data = (finger & 0b111) | ((is_palmar & 0b1) << 3) | ((joint & 0b11) << 4)
    
    def from_array(arr):
        arr = arr.astype(int)
        result = np.zeros((arr.shape[0], 3), dtype=int)
        result[:, 0] = arr % 8
        result[:, 1] = (arr >> 3) % 2
        result[:, 2] = (arr >> 4) % 4
        return result

    # Get finger field
    def get_finger(self):
        return Finger(self.data & 0b111)  # Extract lower 3 bits

    # Get is_palmar field
    def get_is_palmar(self):
        return IsPalmar((self.data >> 3) & 0b1)  # Extract bit 4

    # Get joint field
    def get_joint(self):
        return Joint((self.data >> 4) & 0b11)  # Extract bits 5-6

    # Print binary representation of the structure
    def to_bin(self):
        return f'{self.data:08b}'  # Convert to 8-bit binary string
