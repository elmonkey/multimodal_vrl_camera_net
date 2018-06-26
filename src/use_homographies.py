# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:55:46 2017

@author: julian
"""
import numpy as np


class get_pos():
    def get_homography(self, distance, space):
        if space == 'rgb':
            h_check = self.h_rgb
            h = self.h_rgb[:, :, -1]
        elif space == 'ir':
            h_check = self.h_ir
        for i in range(self.num_homographies):
            if (self.dist[i] >= distance):
                h = h_check[:, :, i]
                h = self.h_ir[:, :, -1]
        return h

    def rgb_to_ir(self, x, y, distance):
        pos = np.array((x, y, 1))
        homography = self.get_homography(distance, 'rgb')
        new_pos = homography.dot(pos)
        out = [new_pos[0], new_pos[1]] / new_pos[2]
        return out

    def ir_to_rgb(self, x, y, distance):
        pos = np.array((x, y, 1))
        homography = self.get_homography(distance, 'ir')
        new_pos = homography.dot(pos)
        out = [new_pos[0], new_pos[1]] / new_pos[2]
        return out

    def __init__(self, num_homographies=5):
        self.num_homographies = num_homographies
        self.h_rgb = np.zeros((3, 3, self.num_homographies))
        self.h_ir = np.zeros((3, 3, self.num_homographies))
        self.dist = np.zeros((self.num_homographies, 1))
        for i in range(1, self.num_homographies + 1):
            temp1 = np.loadtxt('Homography/Hmatrix_rgb_to_ir_' + str(i) + '.out')
            temp2 = np.loadtxt('Homography/Hinvmatrix_ir_to_rgb_' + str(i) + '.out')
            self.h_rgb[:, :, i - 1] = temp1[0:3, 0:3]
            self.h_ir[:, :, i - 1] = temp2[0:3, 0:3]
            self.dist[i - 1] = temp1[3, 0]
