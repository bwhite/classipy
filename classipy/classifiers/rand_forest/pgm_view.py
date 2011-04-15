#!/usr/bin/env python
"""
Usage:
python pgm_view.py mydirwithpgms/
"""
import sys
import numpy as np
import glob
import matplotlib.pyplot as mp


def onclick(event):
    global depth
    if event.xdata == None or event.ydata == None:
        return
    x = int(event.xdata)
    y = int(event.ydata)
    print('[%f, %f] = %d' % (x, y, depth[y, x]))


def onkey(event):
    global path_index
    print(event.key)
    if event.key == 'left':
        path_index -= 1
        if path_index < 0:
            path_index = len(paths) - 1
        depth_from_path()
        ax.imshow(depth)
        mp.draw()
    elif event.key == 'right':
        path_index += 1
        if path_index >= len(paths):
            path_index = 0
        depth_from_path()
        ax.imshow(depth)
        mp.draw()


def depth_from_path():
    global depth
    depth_path = paths[path_index]
    print(depth_path)
    with open(depth_path) as fp:
        depth = fp.read().split('\n', 1)[1]
    depth = np.fromstring(depth, dtype=np.uint16).reshape((480, 640))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    paths = glob.glob(sys.argv[1] + '/*.pgm')
    path_index = 0
    depth = None
    depth_from_path()
    fig = mp.figure()
    ax = fig.add_subplot(111)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    ax.imshow(depth)
    mp.show()
