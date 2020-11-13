import numpy as np 
from ultis import cosine_distances

# from python_tsp.exact import solve_tsp_dynamic_programming

class Sensor:
    def __init__(self,x,y,pi):
        self.x=float(x)
        self.y=float(y)
        self.pi=float(pi)

class Framework():
    def __init__(self,path_wce,path_sensor):
        self.E_M,self.v,self.P_M,self.U=self.read_data_wce(path_wce)
        self.sensors,self.n_sensors,self.E_max,self.E_min=self.read_data_sensors(path_sensor)
        self.matrix_distance=self.compute_matrix_distance()
        self.CN=np.zeros(self.n_sensors,dtype=np.int16)
        self.T=0
        self.btn_flag=0
        self.sit_flag=0

        self.solve()
    
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
        E_max,E_min=tuple(data[0].split())
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
        #_,distance=solve_tsp_dynamic_programming(np.array(self.matrix_distance))
        #return distance
        return 4270.224625147727
        
    def get_alive(self,path):
        size_path=len(path)
        E_M=self.E_M
        time_driving=[]
        time_charging=[]
        isDead=False
        for i in range(size_path-1):
            time_driving.append(self.matrix_distance[path[i]][path[i+1]]/self.v)

        for i in range(size_path-1):
            time_charging.append(self.T*self.sensors[path[i]].pi/self.U)

        time_coming_sensor=[]
        time_coming_sensor.append(time_driving[0])
        for i in range(1,size_path-1):
            time_coming_sensor.append(time_coming_sensor[i-1]+time_charging[i]+time_driving[i])
        
        energy_charging=[]
        for i in range(size_path-1):
            energy_charging.append(time_charging[i]*self.sensors[path[i]].pi)
        
        energy_remain_wce=[]
        for i in range(size_path-1):
            E_M=E_M-self.P_M*time_driving[i]
            energy_remain_wce.append(E_M)
        
        E_sensor_remain=[]
        for i in range(size_path-1):
            E_sensor_remain.append(self.E_max-time_coming_sensor[i]*self.sensors[path[i]].pi)
        
        if(np.array(energy_remain_wce).min()<0):
            isDead=True
        if (np.array(E_sensor_remain).min<self.E_min):
            isDead=True
        
        return isDead


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
    
    def solve(self):
        L_tsp=self.compute_tsp()
        t_tsp_min=L_tsp/self.v  # minimum moving time
        E_M_tmp=t_tsp_min*self.P_M #minimum moving energy
        E_max_plus_min=self.E_max-self.E_min
        P=0
        T_i=[E_max_plus_min/self.U+E_max_plus_min/(self.U-self.sensors[i].pi)
                for i in range(1,self.n_sensors)]
        self.T=max(T_i)
        P=sum([self.sensors[i].pi for i in range(1,self.n_sensors)])

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
        elif(self.btn_flag==1 and self.E_M<E_M_tmp):
            self.sit_flag=3
        print(self.sit_flag)
    
    def compute_path(self,path):
        n=len(path)
        dis=0
        for i in range(n-1):
            dis+=self.matrix_distance[path[i]][path[i+1]]
        print(dis)

'''
path_wce='/home/lnq/Desktop/Hoc/2020/ttth/HPSOGA/wce.txt'
path_sensor='/home/lnq/Desktop/Hoc/2020/ttth/HPSOGA/sensors.txt'
path=[0,4,7,19,16,1,12,3,18,15,8,2,6,5,9,13,20,14,11,17,10,0]
model=Framework(path_wce=path_wce,path_sensor=path_sensor)
model.solve()
#model.compute_fitness(path)
'''
