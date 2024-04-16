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


def rL1(x,y): return abs(x-y).mean()/abs(x).mean();
def rL2(x,y): return np.sqrt( ((x-y)**2).mean()/(x**2).mean() )

class LangevinIntegrator():
    def __init__(self,dim):
        self.dim = dim
    def get_data(self,x0,f,eps,dt,m,T0,T,get_x_bc=None,use_tqdm=True):
        D = []
        x = x0
        if use_tqdm: i_range = tqdm(range(round(T/dt)))
        else: i_range = range(round(T/dt))
        for i in i_range:
            x = x + f(x)*dt + np.sqrt(2*eps*dt)*np.random.normal(0,1,x.shape)
            if get_x_bc is not None: x = get_x_bc(x)
            if i>=round(T0/dt) and (i-round(T0/dt))%m==0: D.append(x)
        return np.reshape(D,(-1,self.dim))
    def TwoTempering(self,x1,x2,V,grad,eps1,eps2,dt1,dt2,m,T0,T,m_swap):
        def if_swap(delta_V,delta_beta): return np.random.uniform(0,1)<np.minimum(np.exp(delta_V*delta_beta),1)
        D = []
        for i in range(int(T/dt1)):
            if i>T0/dt1 and (i-T0/dt1)%m==0: D.append(x1)
            if i%(int(T/dt1)/100)==0: print(i,int(T/dt1)/100);
            if i>0 and i%m_swap==0 and if_swap(V(x1)-V(x2),1/eps1-1/eps2): tmp=x1+0.; x1=x2+0.; x2=tmp;
            x1 = x1 - grad(x1)*dt1 + np.sqrt(2*eps1*dt1)*np.random.normal(0,1,x1.shape)
            x2 = x2 - grad(x2)*dt2 + np.sqrt(2*eps2*dt2)*np.random.normal(0,1,x2.shape)
        return np.reshape(D,(-1,self.dim))

class FCNN(nn.Module):
    def __init__(self,input_dim=2,output_dim=1,num_hidden=2,hidden_dim=10,act=func.tanh,transform=None):
        super().__init__()
         
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers  = nn.ModuleList([nn.Linear(input_dim,hidden_dim)])
        for _ in range(num_hidden-1): self.layers.append(nn.Linear(hidden_dim,hidden_dim))
        self.act     = act
        self.out     = nn.Linear(hidden_dim,output_dim)
        self.transform = transform
    def forward(self,X):
        if self.transform is not None: X = self.transform(X)
        for layer in self.layers: X = self.act(layer(X))
        Y = self.out(X)
        return Y

class Model_q(FCNN):
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
    
class Model_q_cpu(FCNN):
    def __init__(self,out_act=func.sigmoid,**kwargs):
        super().__init__(**kwargs)
        self.out_act = out_act
    def get_qt(self,X): 
        if torch.is_tensor(X): 
            return super().forward(X).reshape(-1)
        else: 
            X  = torch.tensor(X)
            qt = super().forward(X).reshape(-1)
            return qt.cpu().data.numpy()
    def get_q(self,X): 
        if torch.is_tensor(X): 
            return self.out_act(super().forward(X)).reshape(-1)
        else: 
            X = torch.tensor(X)
            q = self.out_act(super().forward(X)).reshape(-1)
            return q.cpu().data.numpy()
    def get_q_qx(self,X): 
        if torch.is_tensor(X):
            q  = self.get_q(X)
            qx = torch.autograd.grad(q,X,torch.ones_like(q),create_graph=True)[0]
        else:
            X = torch.tensor(X,requires_grad=True)
            q  = self.get_q(X)
            qx = torch.autograd.grad(q,X,torch.ones_like(q),create_graph=True)[0]
            q  = q.cpu().data.numpy()
            qx = qx.cpu().data.numpy()
        return q,qx
class Model_cpu(Model_q_cpu):  # model for q(x) and r(x)
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
            if i_step%n_show_loss==0:
                loss_train,loss_test = self.get_loss(data_train,c1,c2)[:-1],\
                                       self.get_loss(data_test,c1,c2)[:-1]

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

class Metadynamics(object):
    def __init__(self,model,h,w,L=0,R=1,Nr=int(1e4)):
        self.model = model
        self.h     = h; 
        self.w     = w; 
        
        self.r_j     = np.linspace(L,R,Nr+1); 
        self.dr      = (self.r_j[-1]-self.r_j[0])/Nr;
        self.V_j     = np.zeros(dtype=np.float64,shape=(len(self.r_j)))
        self.V_j_der = np.zeros(dtype=np.float64,shape=(len(self.r_j)))
        self.store_X = []
    def get_j_from_r(self,r): return np.int_((r-self.r_j[0])/self.dr)
    def get_j_from_x(self,x): r = self.model.get_r(x); return self.get_j_from_r(r)
    def get_V(self,x):
        J = self.get_j_from_x(x)
        return np.hstack([self.V_j[j] for j in J])
    def get_dV(self,x):
        r,r_x = self.model.get_r_rx(x); J = self.get_j_from_r(r)
        V_j_der = np.vstack([self.V_j_der[j] for j in J])
        return V_j_der*r_x
    def update_V_dV(self,x):
        self.store_X.append(x[0])
        r = self.model.get_r(x)[0];
        self.V_j = self.V_j + self.h*np.exp(-(self.r_j-r)**2/2/self.w/self.w)
        self.V_j_der = self.V_j_der + self.h*np.exp(-(self.r_j-r)**2/2/self.w/self.w)*(-(self.r_j-r)/self.w/self.w)    
    def re_init(self,):
        self.V_j = np.zeros(dtype=np.float64,shape=(len(self.r_j)))
        self.V_j_der = np.zeros(dtype=np.float64,shape=(len(self.r_j)))
        self.store_X = []    
    def show_meta(self,show_distr,fig_name=None):
        # show V(r), data points, 
        fig,ax = plt.subplots(1,2,figsize=(4,2),constrained_layout=True);
        ax[0].plot(self.r_j,-self.V_j+self.V_j.max()); ax[0].set_xlabel('$r$'); 
        ax[0].set_title(r'$F_r$')

        X = np.vstack(self.store_X); 
        show_distr(X,ax[1])

        if fig_name is not None: 
            plt.savefig(fig_name,dpi=200)
            plt.close(fig)
            return
        plt.show()       
    def perform(self,dV,x,dt,eps,N,N_add,show_distr,show_freq=.1,fig_name=None,get_x_bc=None,use_tqdm=False):
        if use_tqdm: step_range = tqdm(range(N))
        else: step_range = range(N)
        for step in step_range:
            force = -dV(x) - self.get_dV(x)
            xk = x+0.
            x  = x + force*dt + np.sqrt(2*eps*dt)*np.random.normal(0,1,x.shape)
            if get_x_bc is not None: x = get_x_bc(x)
            if step%N_add==0: self.update_V_dV(xk);
            if step%(N*show_freq)==0: 
                if fig_name is not None: 
                    self.show_meta(show_distr,fig_name+'_%d'%step)
                else:
                    self.show_meta(show_distr)       
class Metadynamics_Extend(Metadynamics): # F(x) = F(q(x))
    def __init__(self,eps,**kwargs):
        super().__init__(**kwargs)
        self.eps     = eps
    def get_F_from_q(self,q):
        r,rq = self.model.fn_r(q),self.model.fn_rq(q)
        J    = self.get_j_from_r(r)
        Fr   = np.hstack([self.V_j[j] for j in J])
        return -Fr-self.eps*np.log(rq)
    def get_F(self,X):
        return self.get_F_from_q(self.model.get_q(X))        
    def get_dF(self,X):
        q,qx     = self.model.get_q_qx(X)
        r,rq,tmp = self.model.fn_r(q),self.model.fn_rq(q),self.model.fn_drrdr(q)
        J        = self.get_j_from_r(r)
        Fr_r     = np.vstack([self.V_j_der[j] for j in J])
        return (-Fr_r*rq.reshape(-1,1)-self.eps*tmp.reshape(-1,1))*qx
    def show_meta(self,show_distr,fig_name=None):
        # show V(r), data points, 
        fig,ax = plt.subplots(1,3,figsize=(6,2),constrained_layout=True);
        ax[0].plot(self.r_j,-self.V_j+self.V_j.max()); ax[0].set_xlabel('$r$'); 
        ax[0].set_title(r'$F_r$')

        X = np.vstack(self.store_X); 
        show_distr(X,ax[1])
        
        q_j = self.get_F_from_q(self.r_j[1:-1]); 
        ax[2].plot(self.r_j[1:-1],q_j-q_j.min()); ax[2].set_xlabel('$q$');
        ax[2].set_title(r'$F_q$')

        if fig_name is not None: 
            plt.savefig(fig_name,dpi=200)
            plt.close(fig)
            return
        plt.show()      