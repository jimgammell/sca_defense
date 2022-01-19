import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from scaaml.model import load_model_from_disk

def PretrainedDiscriminator(byte, attack_pt):
    if not((type(byte)==int) and (0 <= byte < 16)):
        raise ValueError("Invalid byte: {}. Must be integer in [0, 15].".format(byte))
    if not(attack_pt in ['key', 'sub_bytes_in', 'sub_bytes_out']):
        raise ValueError("Invalid attack point: {}. Must be one of \"key\", \"sub_bytes_in\", \"sub_bytes_out\".".format(attack_pt))
    model = load_model_from_disk(
        'models/stm32f415-tinyaes-cnn-v10-ap_%s-byte_%d-len_20000'%(attack_pt, byte))
    model.summary()
    return model