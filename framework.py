import numpy as np 
from ultis import *
from hposoga import *

class Sensor:
    def __init__(self,x,y,pi):
        self.x=float(x)
        self.y=float(y)
        self.pi=float(pi)

class framework():
    def __init__(self,path_wce,path_sensor):
        self.E_M,self.v,self.P_M,self.U=self.read_data_wce(path_wce)
        self.sensors,self.n_sensors,self.E_max,self.E_min=self.read_data_sensors(path_sensor)
        self.matrix_distance=self.compute_matrix_distance()
        self.CN=np.zeros(self.n_sensors,dtype=np.int16)
        self.T=None
        self.btn_flag=0
        self.sit_flag=0
    
    def read_data_wce(self,path_data_wce):
        '''
        read data of wce
        '''
        with open(path_data_wce,'r') as f:
            data=f.read().splitlines()
        
        E_M,v,P_M,U=tuple(data)
        return float(E_M),float(v),float(P_M),float(U)

    def read_data_sensors(self,path_sensor):
        with open(path_sensor,'r') as f:
            data=f.read().splitlines()
        n_sensors=len(data)-1
        E_max,E_min=tuple(data[0])
        sensors=[]
        for i in range(1,n_sensors+1):
            x,y,pi=tuple(data[i].split())
            sensors.append(Sensor(x,y,pi))
        
        return sensors,n_sensors,float(E_max),float(E_min)
    
    def compute_matrix_distance(self):
        matrix_distance=np.zeros((self.n_sensors,self.n_sensors))
        for i in range(self.n_sensors-1):
            for j in range(i+1, self.n_sensors):
                v1=np.array([self.sensors[i].x,self.sensors[i].y])
                v2=np.array([self.sensors[j].x,self.sensors[j].y])
                distance=cosine_distances(v1,v2)
                matrix_distance[i][j]=distance
                matrix_distance[j][i]=distance

        return matrix_distance
    
    def compute_tsp(self):
        '''
        Compute the minimum Hamilton cycle length Ltsp of all sensor nodes
        '''
        f_best=np.inf
        c_min=np.inf
        check=np.zeros(self.n_sensors,dtypr=np.int8)  #0|1
        x=np.zeros(self.n_sensors,dtype=np.int16)

        def TSP(k,f_tmp):
            for i in range(1,self.n_sensors):
                if(check[i]==0):
                    x[k]=i
                    check[i]=1
                    f_tmp+=self.matrix_distance[x[k-1]][x[k]]
                    if(k==self.n_sensors-1):
                        if(f_best<f_tmp+self.matrix_distance[x[k]][0]):
                            f_best=f_tmp+self.matrix_distance[x[k]][0]
                    else:
                        if(f_tmp+(self.n_sensors-k)*c_min<f_best):
                            TSP(k+1,f_tmp)
                    check[i]=0
                    f_tmp-=self.matrix_distance[x[k-1]][x[k]]
        
        def solve():
            x[0]=0
            k=1
            f_tmp=0
            check[0]=1
            for i in range(1,self.n_sensors):
                x[k]=i
                check[i]=1
                f_tmp+=self.matrix_distance[x[k-1]],[x[k]]
                TSP(k+1,f_tmp)
                check[i]=0
                f_tmp-=self.matrix_distance(x[k-1],x[k])


        for i in range(self.n_sensors):
            self.matrix_distance[i][i]=np.inf
        
        c_min=min(self.matrix_distance)
        solve()        
        return f_best
    
    def compute_fitness(self,path):
        def time_driving(path):
            n=len(path)
            L_m=0
            for i in range(n-1):
                L_m+=self.matrix_distance[path[i]][path[i+1]]
            return L_m/self.v
        
        def time_charging(path):
            time_charging=0
            for i in path:
                time_charging+=self.sensors[i].pi
            time_charging=time_charging*self.T/self.U
            return time_charging
        
        t_vac=self.T-time_charging(path)-time_driving(path)
        fitness=t_vac/self.T 
        return fitness
    
    def solve(self):
        L_tsp=self.compute_tsp()
        t_tsp_min=L_tsp/self.v  # minimum moving time
        E_M_tmp=t_tsp_min*self.P_M #minimum moving energy
        E_max_plus_min=self.E_max-self.E_min
        P=0
        for i in range(1,self.n_sensors):
            T_i=E_max_plus_min/self.U+E_max_plus_min/(self.U-self.sensors[i].pi)
            if(self.T<=T_i):
                self.T=T_i
            P+=self.sensors[i].pi
    
        for i in range(1,self.n_sensors):
            t_i_vac=self.T-t_tsp_min-P*self.T/self.U
            if(t_i_vac<0):
                self.btn_flag=1
                n_i=(self.T*self.sensors[i].pi*(self.U-self.sensors[i].pi))/(self.U*E_max_plus_min)
                n_i=int(np.ceil(n_i))
                self.CN[i]=n_i
        
        if(self.btn_flag==1 and self.E_M>E_M_tmp):
            self.sit_flag=1
        elif(self.btn_flag==0 and self.E_M<E_M_tmp):
            self.sit_flag=2
        else:
            self.sit_flag=3
        
        # HPSOGA()
