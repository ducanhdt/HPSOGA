import numpy as np 
from ultis import cosine_distances
import copy
from python_tsp.heuristics import solve_tsp_simulated_annealing
import os

class Sensor:
    def __init__(self,x,y,pi):
        self.x=float(x)
        self.y=float(y)
        self.pi=float(pi)*2

class Framework():
    def __init__(self,path_wce,path_sensor):
        self.E_MC,self.v,self.P_M,self.U=self.read_data_wce(path_wce)
        self.sensors,self.n_sensors,self.E_remain_T=self.read_data_sensors(path_sensor)
        self.matrix_distance=self.compute_matrix_distance()
        self.CN=np.zeros(self.n_sensors,dtype=np.int16)
        self.T=0
        self.btn_flag=0
        self.sit_flag=0
        self.E_max=10800
        self.E_min=540
        self.n_sensors_encode=self.n_sensors
        self.solve()

    def read_data_wce(self,path_data_wce):
        '''
        read data of wce
        '''
        with open(path_data_wce,'r') as f:
            data=f.read().splitlines()

        E_MC,v,P_M,U=tuple(data)
        return float(E_MC),float(v),float(P_M),float(U)

    def read_data_sensors(self,path_sensor):
        with open(path_sensor,'r') as f:
            data=f.read().splitlines()
        n_sensors=len(data)
        sensors=[]
        E_remain_T=[]
        x,y=tuple(data[0].split())
        sensors.append(Sensor(x,y,0))
        for i in range(1,n_sensors):
            x,y,pi,e_remain=tuple(data[i].split())
            sensors.append(Sensor(x,y,pi))
            E_remain_T.append(e_remain)
        
        return sensors,n_sensors,E_remain_T
    
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
        _,distance=solve_tsp_simulated_annealing(np.array(self.matrix_distance))
        return distance
        
    def get_alive(self,path):
        path_tmp=[0]+path
        size_path=len(path_tmp)
        time_driving=[]
        n_dead=0
        check_dead=[1]*(size_path-1)
        time_max=72000
        E_mc=self.E_MC
        E_remain=copy.copy(self.E_remain_T)
        E_remain=[0]+E_remain
        for i in range(size_path-1):
            time_driving.append(self.matrix_distance[path_tmp[i]][path_tmp[i+1]]/self.v)
        
        time_coming=0
        count_cycle=0
        total_time=0
        time_end_charging_each_node=[0]*size_path
        while(total_time<=time_max):
            time_coming=0
            for i in range(1,size_path):
                if(total_time>time_max):
                    break
                time_coming+=time_driving[i-1]
                total_time+=time_driving[i-1]
        
                if(check_dead[i-1]==0):
                    new_time=float(self.E_max-self.E_min)/(self.U-self.sensors[i].pi)
                    time_coming+=new_time
                    total_time+=new_time
                    continue

                E_remain[i]=float(E_remain[i])-float(self.sensors[i].pi)*time_coming
                E_mc=E_mc-self.P_M*time_driving[i-1]
                if(E_mc<=0):
                    for j in range(i-1,size_path-1):
                        check_dead[j]=0
                    break
                if(E_remain[i]<self.E_min):
                    check_dead[i-1]=0
                    new_time=float(self.E_max-self.E_min)/(self.U-self.sensors[i].pi)
                    time_coming+=new_time
                    total_time+=new_time
                else:
                    new_time=(self.E_max-E_remain[i])/(self.U-self.sensors[i].pi)
                    time_coming+=new_time
                    total_time+=new_time
                    time_end_charging_each_node[i]=total_time
                    E_mc=E_mc-(self.E_max-E_remain[i])
                    E_remain[i]=self.E_max
            
            total_time+=(self.matrix_distance[path[-1]][0])/self.v
            assert (self.T>time_coming)
            for i in range(1,size_path):
                # total_time+=+self.T-time_coming
                E_remain[i]-=(total_time-time_end_charging_each_node[i])*self.sensors[i].pi
            n_dead=size_path-1-sum(check_dead)
            count_cycle+=1
            print('CYCLE {}: '.format(count_cycle) )
            print('So luong nut chet: ',n_dead)
            print('Thoi gian hien tai:', total_time)
            print('Nang luong con lai cua moi nut: ',E_remain)
            print('thoi gian ket thuc moi node: ',time_end_charging_each_node)
            print('Trang thai cua moi nut: ',check_dead)
            print('Nang luong con lai cua WCE: ',max(E_mc,0))
            print('time max ', time_max)
            print('-------------------------------------------------------------------------------------')
            E_mc=self.E_MC
        print(path_tmp)
    def compute_fitness(self,path):
        def time_driving(path):
            n=len(path)
            L_m=0
            for i in range(n-1):
                L_m+=self.matrix_distance[path[i]][path[i+1]]
            L_m+=self.matrix_distance[0][path[0]]+self.matrix_distance[path[n-1]][0]
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
    
    def encode(self):
        self.n_sensors_encode=self.n_sensors+sum(self.CN)
        self.list_sensors_encode=[0]*self.n_sensors_encode
        count_CN=0
        for i in range(self.n_sensors):
            self.list_sensors_encode[i]=i
            if(self.CN[i]!=0):
                for j in range(self.CN[i]):
                    self.list_sensors_encode[self.n_sensors+count_CN+j-1]=i
                count_CN+=self.CN[i]
        
        matrix_distance_encode=np.zeros((self.n_sensors_encode,self.n_sensors_encode))
        for i in range(self.n_sensors_encode-1):
            for j in range(self.n_sensors_encode):
                v1=np.array([self.sensors[self.list_sensors_encode[i]].x,\
                            self.sensors[self.list_sensors_encode[i]].y])
                v2=np.array([self.sensors[self.list_sensors_encode[j]].x,\
                            self.sensors[self.list_sensors_encode[j]].y])
                distance=cosine_distances(v1,v2)
                matrix_distance_encode[i][j]=distance
                matrix_distance_encode[j][i]=distance
        
        sensors_encode=[]
        for i in range(self.n_sensors_encode):
            sensors_encode.append(copy.copy(self.sensors[self.list_sensors_encode[i]]))
        self.sensors=sensors_encode
        self.matrix_distance=matrix_distance_encode

    def decode(self,path):
        path_decode=[]
        if(self.n_sensors!=self.n_sensors_encode):
            for i in path:
                path_decode.append(self.list_sensors_encode[i])
        else:
            path_decode=path
        return path_decode
    
    def solve(self):
        L_tsp=self.compute_tsp()
        t_tsp_min=L_tsp/self.v  # minimum moving time
        E_dri_tmp=t_tsp_min*self.P_M #minimum moving energy
        E_max_plus_min=self.E_max-self.E_min
        P=0
        T_i=[E_max_plus_min/self.sensors[i].pi+E_max_plus_min/(self.U-self.sensors[i].pi)
                for i in range(1,self.n_sensors)]
        self.T=max(T_i)
        P=sum([self.sensors[i].pi for i in range(1,self.n_sensors)])

        for i in range(1,self.n_sensors):
            t_i_vac=self.T-t_tsp_min-P*self.T/self.U
            if(t_i_vac<0):
                self.btn_flag=1
                n_i=(self.T*self.sensors[i].pi*(self.U-self.sensors[i].pi))/(self.U*E_max_plus_min)
                n_i=int(np.ceil(n_i))
                if(n_i>1):
                    self.CN[i]=n_i-1
        
        if(self.btn_flag==1 and self.E_MC>E_dri_tmp):
            self.sit_flag=1
        elif(self.btn_flag==0 and self.E_MC<E_dri_tmp):
            self.sit_flag=2
        elif(self.btn_flag==1 and self.E_MC<E_dri_tmp):
            self.sit_flag=3
        
        if(self.sit_flag==1):
            self.encode()
        
        #print(self.sit_flag)
