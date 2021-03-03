#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:43:32 2021

@author: chahe
"""
import numpy as np
from numpy.linalg import inv
import math

#the intersection of two points/lines
def crossproduct(x,y):
    z =np.array([x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]])
    return z

def getHomoTranPoint(Hmat,point):
    Hmat=np.array(Hmat).reshape(len(point),len(point))
    print(Hmat)
    return Hmat@np.array(point)

def getProjOfPoint(Pmat,point):
    Pmat = np.array(Pmat).reshape(3,4)
    data = Pmat@np.array(point)
    return (data[0]/data[2],data[1]/data[2])
    
def getHomoTranLine(Hmat,line):
    Hmat=np.array(Hmat).reshape(3,3)
    return inv(Hmat).transpose()@np.array(line)
 
def getRotationMatrix(angle,isdegree = True):
    angle = math.pi/180*angle
    R_mat=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    return R_mat
    
def getSkewSymmMatrix(vector3d):
    return (np.array([[0,-vector3d[2],vector3d[1]],[vector3d[2],0,-vector3d[0]],[-vector3d[1],vector3d[0],0]]))
 
def getHomoFromPointToLine(Hmat):
    Hmat=np.array(Hmat).reshape(3,3)
    return inv(Hmat).transpose()

def getProjMatFromKR(k,h):
    k_mat=np.array(k).reshape(3,3)
    k_mat =np.hstack((k_mat, np.zeros((3,1))))
    h_mat=np.array(h).reshape(4,4)
    print("K\n",k_mat)
    print("H\n ",h_mat)
    return k_mat@h_mat