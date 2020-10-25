import numpy as np 
from ultis import *
from hposoga import *

class framework():
    def __init__(self,path_data):
        self.x,self.y,self.pi,self.E_max,self.E_min,self.T,self.U,self.n_sensors,self.E_M,self.P_M,self.v=self.read_data(path_data)
        self.matrix_distance=self.compute_matrix_distance()
        self.CN=np.zeros(self.n_sensors,dtype=np.int16)
        self.btn_flag=0
        self.sit_flag=0
    
    def read_data(self,path_data):
        '''
        read data
        '''

    def compute_matrix_distance(self):
        matrix_distance=np.zeros((self.n_sensors,self.n_sensors))
        for i in range(self.n_sensors-1):
            for j in range(i+1, self.n_sensors):
                matrix_distance[i][j]=cosine_distances(v1=np.array([self.x[i],self.y[i]]),
                                                        v2=np.array([self.x[j],self.y[j]]))
        return matrix_distance
    
    def compute_tsp(self):
        '''
        Compute the minimum Hamilton cycle length Ltsp of all sensor nodes
        '''
        return L_tsp

    def solve(sefl):
        L_tsp=self.compute_tsp()
        t_tsp_min=L_tsp/self.v  # minimum moving time
        E_M_tmp=t_tsp_min*self.P_M #minimum moving energy
        for i in range(1,self.n_sensors):
            E_i_tmp=self.E_max[i]-self.E_min[i] 
            T_i=E_i_tmp/self.U+E_i_tmp/(self.U-self.pi[i])
            if(self.T<=T_i):
                self.T=T_i
            self.P+=self.pi[i]
    
        for i in range(1,self.n_sensors):
            t_i_vac=T-t_tsp_min-self.P*self.T/self.U
            if(t_i_vac<0):
                self.btn_flag=1
                n_i=(self.T*self.pi[i]*(self.U-self.pi[i]))/(self.U*(self.E_max[i]-self.E_min[i]))
                n_i=int(np.ceil(n_i))
                self.CN[i]=n_i
        
        if(self.btn_flag==1 and self.E_M>E_M_tmp):
            self.sit_flag=1
        elif(sefl.btn_flag==0 and sefl.E_M<E_M_tmp):
            self.sit_flag=2
        else:
            self.sit_flag=3
        
        HPSOGA()


    




