import numpy as np 
from ultis import cosine_distances
import copy
import os

class Sensor:
    def __init__(self,x,y,pi):
        self.x=float(x)
        self.y=float(y)
        self.pi=float(pi)

class Framework():
    def __init__(self,path_wce,path_sensor,path_driving):
        self.E_MC,self.v,self.P_M,self.U=self.read_data_wce(path_wce)
        self.sensors,self.n_sensors,self.E_remain_T=self.read_data_sensors(path_sensor)
        self.matrix_distance=self.compute_matrix_distance()
        self.T=7200
        self.E_max=10800
        self.E_min=540
        self.path=self.read_path_driving(path_driving)
        
    def read_path_driving(self,path_data_driving):
        with open(path_data_driving,'r') as f:
            path=f.read()
        return list(np.array(path.split(),dtype=np.int))

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
        
    def fobj(self,time_charging):
        size_path=len(self.path)
        path_tmp=[0]+self.path
        time_driving=[]
        n_dead=0
        check_dead=[1]*(size_path-1)
        E_mc=self.E_MC
        E_remain=copy.copy(self.E_remain_T)
        E_remain=[0]+E_remain
        for i in range(size_path-1):
            time_driving.append(self.matrix_distance[path_tmp[i]][path_tmp[i+1]]/self.v)
        
        time_coming=0
        for i in range(1,size_path):
            time_coming+=time_driving[i-1]
            if(time_coming>self.T):
                for j in range(i-1,size_path-1):
                    check_dead[j]=0
                break
            E_remain[i]=float(E_remain[i])-float(self.sensors[i].pi)*time_coming
            E_mc=E_mc-self.P_M*time_driving[i-1]
            if(E_mc<=0):
                for j in range(i-1,size_path-1):
                    check_dead[j]=0
                break
            if(E_remain[i]<self.E_min):
                check_dead[i-1]=0
                time_coming+=time_charging[i]
            else:
                new_time=(self.E_max-E_remain[i])/(self.U-self.sensors[i].pi)
                new_time=min(time_charging[i],new_time)
                time_coming+=time_charging[i]
                E_mc=E_mc-new_time*self.sensors[i].pi            
                        
        n_dead=size_path-1-sum(check_dead)
        return n_dead        
    
    def de(self, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
        dimensions = len(bounds)
        pop = np.random.rand(popsize, dimensions)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([self.fobj(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        for i in range(its):
            for j in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
                mutant = np.clip(a + mut * (b - c), 0, 1)
                cross_points = np.random.rand(dimensions) < crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial * diff
                f = self.fobj(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
            print(best, fitness[best_idx])
        


path_wce='/home/lnq/Desktop/Hoc/2020/ttth/DE/wce.txt'
path_sensor='/home/lnq/Desktop/Hoc/2020/ttth/DE/u75_01_simulated.txt'
path_driving='/home/lnq/Desktop/Hoc/2020/ttth/DE/driving.txt'

bounds=[(-10,10)]*75
model=Framework(path_wce,path_sensor,path_driving)
model.de(bounds,its=10)

    
