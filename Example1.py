import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch 
from torch import nn
torch.set_default_dtype(torch.float64)
import torch.nn.functional as func
from tqdm.notebook import tqdm

class Mueller_System(object):
    dim   = 10
    xrange=[-1.5,1]
    yrange=[-0.5,2]
    A     = np.array([-0.558, 1.441]+[0]*8, dtype=np.float64)
    B     = np.array([0.623, 0.028]+[0]*8, dtype=np.float64)
    r     = 0.1
    R     = 0.12
    
    aa    = [-1,-1,-6.5,0.7]
    bb    = [0,0,11,0.6]
    cc    = [-10,-10,-6.5,0.7]
    AA    = [-200,-100,-170,15]
    XX    = [1,0,-0.5,-1]
    YY    = [0,0.5,1.5,1]
    sigma = 0.05
    
    def __init__(self,): 
        pass
    def get_V_2D(self,X):
        X = np.array(X)
        ee = 0
        if len(X.shape)==2:
            for j in range(4):
                ee =  ee + self.AA[j]*np.exp(self.aa[j]*(X[:,0]-self.XX[j])**2+
                                             self.bb[j]*(X[:,0]-self.XX[j])*(X[:,1]-self.YY[j])+
                                             self.cc[j]*(X[:,1]-self.YY[j])**2)
            ee += 9*np.sin(2*5*np.pi*X[:,0])*np.sin(2*5*np.pi*X[:,1])
        else:
            for j in range(4):
                ee =  ee + self.AA[j]*np.exp(self.aa[j]*(X[0]-self.XX[j])**2+
                                             self.bb[j]*(X[0]-self.XX[j])*(X[1]-self.YY[j])+
                                             self.cc[j]*(X[1]-self.YY[j])**2)
            ee += 9*np.sin(2*5*np.pi*X[0])*np.sin(2*5*np.pi*X[1])
        return ee
    def get_V(self,X):
        X = np.array(X)
        if len(X.shape)==2:
            ee = self.get_V_2D(X[:,:2])
            for i in range(2,self.dim): ee += X[:,i]**2/2/self.sigma**2
        else:
            ee = self.get_V_2D(X[:2])
            for i in range(2,self.dim): ee += X[i]**2/2/self.sigma**2
        return ee
    def get_dV(self,X):
        if len(np.shape(X))==1:
            gg = np.zeros(shape=(self.dim,),dtype=np.float64)
            for j in range(4):
                ee = self.AA[j]*np.exp(self.aa[j]*(X[0]-self.XX[j])**2+
                                       self.bb[j]*(X[0]-self.XX[j])*(X[1]-self.YY[j])+
                                       self.cc[j]*(X[1]-self.YY[j])**2)
                gg[0] = gg[0] + (2*self.aa[j]*(X[0]-self.XX[j])+
                                   self.bb[j]*(X[1]-self.YY[j]))*ee
                gg[1] = gg[1] + (  self.bb[j]*(X[0]-self.XX[j])+
                                 2*self.cc[j]*(X[1]-self.YY[j]))*ee
            gg[0] += 9*2*5*np.pi*np.cos(2*5*np.pi*X[0])*np.sin(2*5*np.pi*X[1])
            gg[1] += 9*2*5*np.pi*np.sin(2*5*np.pi*X[0])*np.cos(2*5*np.pi*X[1])
            for i in range(2,self.dim): gg[i] = X[i]/self.sigma**2
        else:
            gg = np.zeros(shape=(X.shape),dtype=np.float64)
            for j in range(4):
                ee = self.AA[j]*np.exp(self.aa[j]*(X[:,0]-self.XX[j])**2+
                                       self.bb[j]*(X[:,0]-self.XX[j])*(X[:,1]-self.YY[j])+
                                       self.cc[j]*(X[:,1]-self.YY[j])**2)
                gg[:,0] = gg[:,0] + (2*self.aa[j]*(X[:,0]-self.XX[j])+
                                   self.bb[j]*(X[:,1]-self.YY[j]))*ee
                gg[:,1] = gg[:,1] + (  self.bb[j]*(X[:,0]-self.XX[j])+
                                 2*self.cc[j]*(X[:,1]-self.YY[j]))*ee
            gg[:,0] += 9*2*5*np.pi*np.cos(2*5*np.pi*X[:,0])*np.sin(2*5*np.pi*X[:,1])
            gg[:,1] += 9*2*5*np.pi*np.sin(2*5*np.pi*X[:,0])*np.cos(2*5*np.pi*X[:,1])
            for i in range(2,self.dim): gg[:,i] = X[:,i]/self.sigma**2
        return gg
    def get_force(self,X): return -self.get_dV(X)   
    def get_phi(self,X): return X[:,:2]
    def IsInA(self,X): return np.linalg.norm(X[:,:2]-self.A[:2],axis=-1)<self.r
    def IsInB(self,X): return np.linalg.norm(X[:,:2]-self.B[:2],axis=-1)<self.r
    def get_q_MC(self,X,eps,dt=1e-5,n_trajs=1000):
        q = []
        for x0 in np.reshape(X,(-1,self.dim)):
            n_A=0; n_B=0;
            for traj in range(n_trajs):
                x = x0 + 0.
                for i in range(int(1e8)):
                    if self.IsInA(x): n_A+=1; break
                    if self.IsInB(x): n_B+=1; break
                    x = x+self.get_force(x)*dt+np.sqrt(2*eps*dt)*np.random.normal(0,1,x.shape)
            print(n_A,n_B,n_B/(n_A+n_B))
            q.append(n_B/(n_A+n_B))
        return np.array(q)
    def get_q_MC_Par(self,X,eps,dt=1e-5,n_trajs=1000,use_tqdm=False):
        X_back = X + 0.
        nA,nB = 0,0
        if use_tqdm==True: traj_range = tqdm(range(n_trajs))
        else: traj_range = range(n_trajs)
        for traj in traj_range:
            mask  = np.full(X.shape[0], True)
            X     = X_back + 0.
            for i in range(int(1e8)):
                iA,iB = self.IsInA(X),self.IsInB(X)
                if iA.sum()+iB.sum()==X.shape[0]: break
                mask = mask & (~iA) & (~iB)
                X[mask] = X[mask]+self.get_force(X[mask])*dt+np.sqrt(2*eps*dt)*np.random.normal(0,1,X[mask].shape)
            nA += iA; nB += iB; 
        return nB/n_trajs
