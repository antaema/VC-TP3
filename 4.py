from __future__ import division
import numpy as np
import math
import cv2 
import matplotlib.pyplot as plt
import scipy.spatial.distance as scipy
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib import gridspec
import multiprocessing
from joblib import Parallel, delayed
import sys
from  random import randint 

def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def generateLine(i):
    img = np.zeros((480,480))
    density = randint(4,12)
    direction = (randint(-360,360) + randint(-360,360) ) * math.pi / 180
    lenght = randint(30,480)
    beginx = randint(0,480)
    beginy = randint(0,480)
    line = beginx,beginy,lenght,direction
    v = randint(1,4)
    for i in range(0,lenght):
        y = int(round(beginy + i * density * math.sin(direction)))
        x = int(round(beginx + i * density * math.cos(direction)))
        if x < 480 and y < 480 and x >= 0 and y >= 0:    
            img[y][x] = 255
        for j in range(0,4):
            noisex = randint(-v,v+1) + x
            noisey = randint(-v,v+1) + y
            if noisex < 480 and noisey < 480 and noisex >= 0 and noisey >= 0:
                img[noisey][noisex] = 255
    return img,line


def makeLines(n):
    num_cores = multiprocessing.cpu_count()
    lines = []
    results = Parallel(n_jobs=2)(delayed(generateLine)(i) for i in range(0,n))
    cum = np.zeros((480,480))
    for i in range(0,len(results)):
        cum = np.add(cum,results[i][0])
        # line = beginx,beginy,lenght,direction
        lines.append(results[i][1])

    for i in range(0,480):
        for j in range(0,480):
            if randint(0,100) == 1:
                cum[j][i] = 255

    return np.clip(cum,0,255) ,lines

f = plt.figure()
gs = gridspec.GridSpec(11,10)
gs.update( hspace=1.5, wspace=1.5)
a0 = plt.subplot(gs[:,: -3])
a1 = plt.subplot(gs[:, 7:10])
a0.axis("off")
img,linos = makeLines(7)
img2 = np.uint8(img)
a0.imshow(img,'gray')
accumulator, thetas, rhos = hough_line(img2)
a1.imshow(accumulator, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]]),
plt.show()