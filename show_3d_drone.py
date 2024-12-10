import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import DLT
import tkinter
plt.style.use("seaborn-v0_8")
matplotlib.use('TkAgg')
import cv2

pose_keypoints = np.array([0])

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts

def create_radar(rx):
    rx.scatter(0, 0, marker='o' , c = 'green')
    circle1 = plt.Circle((0, 0), 3, color='b', fill=False)
    circle2 = plt.Circle((0, 0), 7, color='b', fill=False)
    circle3 = plt.Circle((0, 0), 10, color='b', fill=False)
    rx.add_patch(circle1)
    rx.add_patch(circle2)
    rx.add_patch(circle3)
    rx.set_xlim((-10, 10))
    rx.set_ylim((-10, 10))

def visualize_3d(p3ds):

    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    radar = plt.figure()
    rx = radar.add_subplot(111)
    

    for kpts3d in p3ds:
        print(kpts3d[0])
        ax.scatter(xs = kpts3d[0][0], ys = kpts3d[0][1], zs = kpts3d[0][2], marker='o' , c = 'red')

        #uncomment these if you want scatter plot of keypoints and their indices.
        # for i in range(12):
        #     #ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
        #     #ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])
        create_radar(rx)
        rx.scatter(x = kpts3d[0][0]+10, y = kpts3d[0][1]+10, marker='o' , c = 'red')

        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-20, 0)
        ax.set_xlabel('x')
        ax.set_ylim3d(-20, 0)
        ax.set_ylabel('y')
        ax.set_zlim3d(0, 90)
        ax.set_zlabel('z')

        plt.pause(0.1)
        ax.cla()
        rx.cla() 


if __name__ == '__main__':

    p3ds = read_keypoints('kpts_3d_r.dat')
    visualize_3d(p3ds)
