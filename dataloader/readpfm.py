from __future__ import unicode_literals
import re
import numpy as np


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match('^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode('utf-8'))
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data, scale


if __name__ == '__main__':
    img_path = \
        '/media/data/dataset/SceneFlow/driving_frames_cleanpass/15mm_focallength/scene_backwards/fast/left/0100.png'
    disp_path = img_path.replace('driving_frames_cleanpass', 'driving_disparity').replace('png', 'pfm')

    data, scale = readPFM(disp_path)
    dataL = np.ascontiguousarray(data, dtype=np.float32)\

    import matplotlib.pyplot as plt
    plt.imshow(dataL)
    plt.show()