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

class Solvated_dimer(object):   
    def __init__(self,n_atoms,L,h,w,r0,eps,rWCA,): 
        self.n_atoms = n_atoms
        self.L       = L
        self.h       = h
        self.w       = w
        self.r0      = r0
        self.eps     = eps
        self.rWCA    = rWCA
        self.dim     = self.n_atoms*3
        self.density = self.n_atoms/self.L**3
    def wrapvector(self,x): return x - np.round(x/self.L)*self.L
    def wrapdistance(self,x,y): return np.linalg.norm(self.wrapvector(x-y),axis=-1)
    def get_X_bc(self,X): return self.wrapvector(X)
    def get_V1(self,r): return self.h*(1-(r-self.r0-self.w)**2/self.w**2)**2
    def get_V2(self,r): tmp = r**(-6); return (4*self.eps*(tmp-1)*tmp+self.eps)*(r<self.rWCA)
    def get_dV1(self,r): return 4*self.h*(r-self.r0)*(r-self.r0-2*self.w)*(r-self.r0-self.w)/self.w**4
    def get_dV2(self,r): tmp = r**(-6); return (4*self.eps*(-12*tmp+6)*tmp/r)*(r<self.rWCA)
    def get_V(self,X):
        X = np.reshape(X,(-1,self.n_atoms*3))
        # between dimer atoms
        r = self.wrapdistance(X[:,:3],X[:,3:6])
        V = self.get_V1(r)
        # between dimer and solvent
        for i in range(2):
            for j in range(2,self.n_atoms):
                r = self.wrapdistance(X[:,i*3:i*3+3],X[:,j*3:j*3+3])
                V = V + self.get_V2(r)
        # between solvent and solvent
        for i in range(2,self.n_atoms):
            for j in range(i+1,self.n_atoms):
                r = self.wrapdistance(X[:,i*3:i*3+3],X[:,j*3:j*3+3])
                V = V + self.get_V2(r)
        return V
    def get_dV(self,X):
        X = np.reshape(X,(-1,self.n_atoms*3))
        dV = np.zeros(dtype=np.float64,shape=(X.shape))
        # between dimer atoms
        r = self.wrapdistance(X[:,:3],X[:,3:6])
        dV[:, :3] = (self.get_dV1(r)/r).reshape(-1,1)*self.wrapvector(X[:,:3]-X[:,3:6])
        dV[:,3:6] = -dV[:, :3]
        # between dimer and solvent
        for i in range(2):
            for j in range(2,self.n_atoms):
                r   = self.wrapdistance(X[:,i*3:i*3+3],X[:,j*3:j*3+3])
                tmp = (self.get_dV2(r)/r).reshape(-1,1)*self.wrapvector(X[:,i*3:i*3+3]-X[:,j*3:j*3+3])
                dV[:,i*3:i*3+3] = dV[:,i*3:i*3+3] + tmp
                dV[:,j*3:j*3+3] = dV[:,j*3:j*3+3] - tmp
        # between solvent and solvent
        for i in range(2,self.n_atoms):
            for j in range(i+1,self.n_atoms):
                r   = self.wrapdistance(X[:,i*3:i*3+3],X[:,j*3:j*3+3])
                tmp = (self.get_dV2(r)/r).reshape(-1,1)*self.wrapvector(X[:,i*3:i*3+3]-X[:,j*3:j*3+3])
                dV[:,i*3:i*3+3] = dV[:,i*3:i*3+3] + tmp
                dV[:,j*3:j*3+3] = dV[:,j*3:j*3+3] - tmp
        return dV
    def get_a_state_mesh(self,):
        x = np.zeros(dtype=np.float64,shape=(self.n_atoms*3,))
        num_per_dim = np.int_(np.ceil(self.n_atoms**(1/3)))
        for i in range(self.n_atoms):
            iz = np.int_(i/num_per_dim**2)
            ixy = i%(num_per_dim**2)
            iy = np.int_(ixy/num_per_dim)
            ix = ixy%num_per_dim
            x[i*3]   = -self.L/2+ix*(self.L/num_per_dim)
            x[i*3+1] = -self.L/2+iy*(self.L/num_per_dim)
            x[i*3+2] = -self.L/2+iz*(self.L/num_per_dim)
        return x
    def get_bond_length(self,X): return self.wrapdistance(X[:,:3],X[:,3:6])
    def IsInA(self,X): return self.get_bond_length(X)<=self.r0
    def IsInB(self,X): return self.get_bond_length(X)>=self.r0+2*self.w
    def get_force(self,X): return -self.get_dV(X)    
    def get_q_MC_Par(self,X,eps,dt=1e-4,n_trajs=1000,use_tqdm=True):
        X_back = X + 0.
        nA,nB = 0,0
        if use_tqdm: traj_range=tqdm(range(n_trajs))
        else: traj_range=range(n_trajs)
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

class Schnet(nn.Module):
    def __init__(self,get_wrapvec,rbf_centers,rbf_gap,n_atoms,input_dim,
                 n_fea=64,act=func.tanh,n_interactions=3,out_act=None):
        super().__init__()
        
        self.get_wrapvec    = get_wrapvec
        self.n_fea          = n_fea
        self.n_atoms        = n_atoms
        self.act            = act
        self.out_act        = out_act
        self.n_interactions = n_interactions
        self.n_rbf          = len(rbf_centers)
        self.rbf_centers    = rbf_centers
        self.rbf_gap        = rbf_gap
        self.input_dim      = input_dim

        self.v1    = torch.nn.Parameter(torch.tensor(
            np.random.normal(0,np.sqrt(1/np.sqrt(self.n_fea)),self.n_fea)))
        self.v2    = torch.nn.Parameter(torch.tensor(
            np.random.normal(0,np.sqrt(1/np.sqrt(self.n_fea)),self.n_fea)))
        
        self.idx_i = np.hstack([[k]*(self.n_atoms-1) for k in range(self.n_atoms)])
        self.idx_j = np.hstack([list(range(0,k,1))+list(range(k+1,self.n_atoms,1)) for k in range(self.n_atoms)])
        
        self.iblock  = nn.ModuleList([])
        for _ in range(self.n_interactions): 
            self.iblock.extend([nn.Linear(self.n_fea,self.n_fea),
                                nn.Linear(self.n_rbf,self.n_fea),
                                nn.Linear(self.n_fea,self.n_fea),
                                nn.Linear(self.n_fea,self.n_fea),
                                nn.Linear(self.n_fea,self.n_fea),])
        self.dense1 = nn.Linear(self.n_fea,self.n_fea//2)
        self.dense2 = nn.Linear(self.n_fea//2,1)      
    def forward_(self,r):
        n_inputs = r.shape[0]
        v1v2     = torch.repeat_interleave(torch.stack([self.v1,self.v2]),
                                           torch.tensor([2,self.n_atoms-2]).cuda(),axis=0)
        X        = torch.repeat_interleave(torch.reshape(v1v2,(1,-1)),n_inputs,axis=0)
        X        = torch.reshape(X,(-1,self.n_atoms,self.n_fea))
        
        r   = torch.reshape(r,(-1,self.n_atoms,3))
        r_i = r[:,self.idx_i]
        r_j = r[:,self.idx_j]
        dij = torch.norm(self.get_wrapvec(r_i-r_j),dim=-1,keepdim=True) # (-1,32*31,1)
#         print(dij.shape,self.rbf_centers.shape,self.rbf_gap.shape)
        rbf = torch.exp(-(dij-self.rbf_centers)**2/self.rbf_gap)          # (-1,32*31,n_rbf)
        
        k=0
        for _ in range(self.n_interactions):
            X_ = torch.reshape(X,(-1,self.n_fea))
            X_ = self.iblock[k](X_); k=k+1;
            X_ = torch.reshape(X_,(-1,self.n_atoms,self.n_fea))

            h = torch.reshape(rbf,(-1,self.n_rbf))
            h = self.iblock[k](h); k=k+1; h = self.act(h)
            h = self.iblock[k](h); k=k+1; h = self.act(h)
            W = torch.reshape(h,(-1,self.n_atoms*(self.n_atoms-1),self.n_fea))  

#             v = W*tf.gather(X_,self.idx_j,axis=1)
            v = W*X_[:,self.idx_j]
            v = torch.sum(torch.reshape(v,(-1,self.n_atoms,self.n_atoms-1,self.n_fea)),axis=2)
            v = torch.reshape(v,(-1,self.n_fea))
            v = self.iblock[k](v); k=k+1; v = self.act(v)
            v = self.iblock[k](v); k=k+1;
            v = torch.reshape(v,(-1,self.n_atoms,self.n_fea))
            
            X = X + v

        X = torch.reshape(X,(-1,self.n_fea))
        X = self.dense1(X); X = self.act(X)
        X = self.dense2(X)
        X = torch.reshape(X,(-1,self.n_atoms))
        Y = torch.sum(X,axis=-1,keepdim=True) 
        
        del X,v,h,W,X_,rbf,dij,v1v2,r,r_i,r_j
        return Y
    def forward(self,X,unit_len=100):
        q = []
        I = int(np.ceil(len(X)/unit_len))
        for i in range(I):
#             torch.cuda.empty_cache()
            X_sub  = X[i*unit_len:(i+1)*unit_len]
            if I>1:
                with torch.no_grad():
                    q.append(self.forward_(X_sub))
            else:
                q.append(self.forward_(X_sub))
        return torch.vstack(q)
    
class Model_q(Schnet):
    def __init__(self,out_act=func.sigmoid,**kwargs):
        super().__init__(**kwargs)
        self.out_act = out_act
    def get_qt(self,X): 
        if torch.is_tensor(X): 
            return super().forward(X).reshape(-1)
        else: 
            X  = torch.tensor(X).cuda()
            qt = super().forward(X).reshape(-1)
            return qt.cpu().data.numpy()
    def get_q(self,X): 
        if torch.is_tensor(X): 
            return self.out_act(super().forward(X)).reshape(-1)
        else: 
            X = torch.tensor(X).cuda()
            q = self.out_act(super().forward(X)).reshape(-1)
            return q.cpu().data.numpy()
    def get_q_qx(self,X): 
        if torch.is_tensor(X):
            q  = self.get_q(X)
            qx = torch.autograd.grad(q,X,torch.ones_like(q),create_graph=True)[0]
        else:
            X = torch.tensor(X,requires_grad=True).cuda()
            q  = self.get_q(X)
            qx = torch.autograd.grad(q,X,torch.ones_like(q),create_graph=True)[0]
            q  = q.cpu().data.numpy()
            qx = qx.cpu().data.numpy()
        return q,qx
class Model(Model_q):  # model for q(x) and r(x)
    def __init__(self,n=10,**kwargs):
        super().__init__(**kwargs)
        self.n = n
    def fn_r(self,q): 
        q_ = q**(1/self.n); 
        return q_/((1-q)**(1/self.n)+q_)
    def fn_rq(self,q): 
        tmp=(1-q)**(1/self.n)+q**(1/self.n); 
        return 1/self.n/tmp**2/(q*(1-q))**(1-1/self.n)
    def fn_drrdr(self,q): 
        t1,t2 = (1-q)**(1/self.n),q**(1/self.n)
        t = t1+t2
        return (-t1+t2-self.n*(2*q-1)*t)/(self.n*(q-1)*q*t)
    def get_r(self,x): 
        q = super().get_q(x); 
        return self.fn_r(q)
    def get_rx(self,x): 
        q,qx = super().get_q_qx(x); 
        return self.fn_rq(q).reshape(-1,1)*qx
    def get_r_rx(self,x): 
        q,qx = super().get_q_qx(x); 
        return self.fn_r(q),self.fn_rq(q).reshape(-1,1)*qx
    
####. Disable showing loss and set torch.cuda.empty_cache() every step
    
class Solver():
    def __init__(self,model,q0=-5,q1=5,unit_len=int(1e5)): 
        self.model    = model
        self.q0       = q0
        self.q1       = q1
        self.unit_len = unit_len
    def sample_batch(self,data,batch_size):
        batch = []
        for X in data:
            if len(X)>0:
                idx = random.sample(range(len(X)),  min(batch_size,len(X)))
                batch.append(X[idx])
            else:
                batch.append([])
        return batch
    def get_r(self,X,coef):
        r = 0
        I = int(np.ceil(len(X)/self.unit_len))
        for i in range(I):
            X_sub    = X[i*self.unit_len:(i+1)*self.unit_len]
            coef_sub = coef[i*self.unit_len:(i+1)*self.unit_len]
            qx       = self.model.get_q_qx(X_sub)[1]
            r        = r + torch.sum(torch.sum(qx**2,axis=-1)*coef_sub)
        return r/len(X)
    def get_LAB(self,X_A,X_B): 
        L_A = 0
        if len(X_A)>0: 
            I = int(np.ceil(len(X_A)/self.unit_len))
            for i in range(I):
                X_sub  = X_A[i*self.unit_len:(i+1)*self.unit_len]
                qt_sub = self.model.get_qt(X_sub)
                q_sub  = self.model.out_act(qt_sub)
                L_A    = L_A + torch.sum(q_sub**2) + torch.sum(func.relu(qt_sub-self.q0)**2)
            L_A = L_A/len(X_A)
        L_B = 0
        if len(X_B)>0:
            I = int(np.ceil(len(X_B)/self.unit_len))
            for i in range(I):
                X_sub  = X_B[i*self.unit_len:(i+1)*self.unit_len]
                qt_sub = self.model.get_qt(X_sub)
                q_sub  = self.model.out_act(qt_sub)
                L_B    = L_B + torch.sum((1-q_sub)**2) + torch.sum(func.relu(self.q1-qt_sub)**2)
            L_B = L_B/len(X_B)
        return L_A+L_B
    def get_loss(self,data,c1,c2): 
        l1 = 0;  
        if c1>0: 
            X,coef = data[0][:,:self.model.input_dim],data[0][:,-1]
            l1 = self.get_r(X,coef);
        l2 = 0; 
        if c2>0: 
            X_A,X_B = data[1],data[2]
            l2 = self.get_LAB(X_A,X_B);
        loss = c1*l1 + c2*l2
        return l1,l2,loss    
    def train_model(self,data_train,data_test,c1,c2,batch_size,optimizer,n_steps,
                    scheduler=None,n_show_loss=100,terminal_condition=None,error_model1=None,error_model2=None,use_tqdm=True):
        
        if use_tqdm: step_range = tqdm(range(n_steps))
        else: step_range = range(n_steps)
        loss_step = []
        for i_step in step_range:
            torch.cuda.empty_cache()
            if i_step%n_show_loss==0:
#                 loss_train,loss_test = self.get_loss(data_train,c1,c2)[:-1],\
#                                        self.get_loss(data_test,c1,c2)[:-1]
                loss_train,loss_test = [-1],[-1]

                def show_num(x): 
                    if abs(x)<100 and abs(x)>.01: return '%0.5f'%x
                    else: return '%0.2e'%x
                item1 = '%2dk'%np.int_(i_step/1000)
                item2 = 'Loss: '+' '.join([show_num(k) for k in loss_train])
                item3 = ' '.join([show_num(k) for k in loss_test])
                item4 = ''
                if error_model1 is not None:
                    item4 = 'Error: '+show_num(error_model1(self.model))
                    if error_model2 is not None:
                        item4 = item4 + ' '+show_num(error_model2(self.model))
                print(', '.join([item1,item2,item3,item4]))
                loss_step = loss_step + [i_step] + [float(k) for k in loss_train]\
                                                 + [float(k) for k in loss_train]
            
            data_batch = self.sample_batch(data_train,batch_size)
            loss       = self.get_loss(data_batch,c1,c2)[-1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
            if terminal_condition is not None:
                if terminal_condition(self.model): return
        return loss_step