import numpy as np
import random
import pandas as pd
from ultis import *
from operator import itemgetter
from framework import Framework
import copy

path_wce = "wce.txt"
path_sensor = "data_chi_huong/normal/n25_01_simulated.txt" #"sensors.txt"

class Individual():
  frame=Framework(path_sensor=path_sensor,
                      path_wce=path_wce)
  def __init__(self,path):
    self.path = np.array(path)
    self.fitness = Individual.frame.compute_fitness(path)
    self.length = len(path)
    self.vec = [0 for _ in range(self.length)]
    self.path_to_position()
    self.pbest = self.position
    self.pbest_fitness = self.fitness
    
  def path_to_position(self):
    path =list(self.path)
    position = []
    l=len(path)
    # path = np.array(path)
    for i in range(len(path)):
      x = path.index(i+1)
      #print(i,x,x-l+1,path[x-l+1])
      position.append(path[x-l+1])
    self.position = position
    return position
    
  def position_to_path(self):
    path = [self.path[0]]
    for i in range(len(self.path)-1):
      path.append(self.position[path[-1]-1])
    path = np.array(path)
    for i in range(len(path)):
      if self.path[i]!= path[i]:
        self.update_path(path)
        break
    return path

  def update_pbest(self,new_position,new_fitness):
    if new_fitness > self.pbest_fitness:
      self.pbest = new_position
      self.pbest_fitness = new_fitness
  def update_path(self,new_path):
    self.path = new_path
    self.fitness = Individual.frame.compute_fitness(self.path)
  def print(self):
    print("path:{}".format(self.path))
    print("position:{}".format(self.position))
    print("fitness:{}".format(self.fitness))
    print("vec:{}".format(self.vec))
    print('--------------')

class HPSOGA():
  """
   implementation of HPSOGA
  """
  def __init__(self,init_instance,population_size,stop_condition=None):
    self.init_instance = init_instance
    self.population_size = population_size
    self.population = self.init_population(population_size)
    self.gbest = self.population[0]
    self.updata_gbest()
    self.k1 = 0.3
    self.k2 = 0.6
    self.c1 = 0.5
    self.c2 = 0.5
    # for i in self.population:
    #   print(i.path)

  def init_population(self,number_instance):
    pop_bag = []
    for i in range(number_instance):
      rnd_sol = self.init_instance.copy()
      random.shuffle(rnd_sol)
      pop_bag.append(Individual(rnd_sol))
    return pop_bag
  
  def updata_gbest(self):
    gbest = self.population[0]
    for indi in self.population:
      if gbest.fitness < indi.fitness:
        gbest = copy.deepcopy(indi)
    if gbest.fitness > self.gbest.fitness:
      self.gbest = copy.deepcopy(gbest)
      print("new best fitness {}".format(self.gbest.fitness))
      print(framework.get_alive(self.gbest.path))
    # print("new gbest fitness: {}".format(gbest.fitness))
  def selectionBest(self):
    new_list = sorted(self.population, key=lambda x: x.fitness, reverse=True)
    self.population = new_list[:self.population_size]

  @staticmethod
  def crossover_operator(father,mother):
    '''
    crossover between par1 vs par2
    '''
    father = father.path 
    mother = mother.path
    start,end = np.sort(np.random.choice(len(father),2,replace=False))
    mother_child = []
    father_child = []
    father_temp = father[start:end]
    mother_temp = mother[start:end]
    
    sm=0
    sf=0
    for i,c in enumerate(mother):
      if c not in father_temp:
        mother_child.append(c)
      elif i>end:
        sm+=1
    for i,c in enumerate(father):
      if c not in mother_temp:
        father_child.append(c)
      elif i>end:
        sf+=1
    
    father_child = father_child[sf:]+father_child[:sf]
    mother_child = mother_child[sm:]+mother_child[:sm]

    father_child = father_child[:start]+list(mother_temp)+father_child[start:]
    mother_child = mother_child[:start]+list(father_temp)+mother_child[start:]

    return Individual(father_child), Individual(mother_child)

  def mutation_operator(self, indi):
      try:
        child = Individual(indi.path)
        # indi.print()
      except:
        print(indi)
      sub_pbest = sub_operate(indi.pbest, indi.position)
      mul_pbest = mul_operate(sub_pbest, self.c1)
      sub_gbest = sub_operate(self.gbest.position, indi.position)
      mul_gbest = mul_operate(sub_gbest, self.c2)
      child.vec = add_operate(mul_pbest, mul_gbest)
      child.position = add_operate_x(child.vec,child.position)
      child.position_to_path()
      child.update_pbest(indi.pbest,indi.fitness)
      return child
  
  def crossover(self):
    '''
    crossover to get new population
    '''
    fitnesses = [indi.fitness for indi in self.population]
    fitness_mean = sum(fitnesses) / len(self.population)
    fitness_max = max(fitnesses)
    i = 0
    for _ in range(self.population_size):
      #choice parent
      i,j = np.random.choice(self.population_size,2,replace=False)
      mother = self.population[i]
      father = self.population[j]

      fitness_bar = max(mother.fitness,father.fitness)

      if fitness_bar > fitness_mean:
          pc = self.k1 - self.k2 * (fitness_max - fitness_bar) / (fitness_max - fitness_mean)
      else:
          pc = self.k1
      rc = random.random()
      if rc < pc:
          child1, child2 = self.crossover_operator(father,mother)
          # if child != -1:
          self.population.append(child1)
          self.population.append(child2)
      i = i + 1
      
      
  def mutation(self):
    new_pop = []
    for i, _ in enumerate(self.population):
      new_pop.append(self.mutation_operator(self.population[i]))


  def evalution(self):
    # print(self.gbest.fitness)
    self.updata_gbest()
    self.selectionBest()
    self.crossover()
    self.mutation()
    self.gbest.print()

if __name__ == "__main__":
  gbest_list = []
  for x in range(1):
    framework = Framework(path_wce= path_wce,path_sensor= path_sensor)
    framework.solve()
    path_init=[i for i in range(1,framework.n_sensors_encode)]
    print(path_init)
    hpsoga=HPSOGA(path_init,100)
    # for i in hpsoga.population:
    #   print(framework.compute_fitness(i.path))
    # neu khong tang sau 50 the he thi dung 
    for _ in range(250):
      print("generation {}:".format(_))
      hpsoga.evalution()
    _,time_charging_each_node,total_charing,t_dri,e_mc_remain = framework.get_alive(hpsoga.gbest.path)
    print("nang luong sac:{}, thoi gian sac: {}".format(total_charing*framework.U,total_charing))
    print("nang luong di chuyen: {}, thoi gian di chuyen: {}".format(t_dri*framework.P_M,t_dri))
    print(time_charging_each_node)
    print("nang luong con lai {}".format(e_mc_remain))
  for i in gbest_list:
    i.print()
    # from os import listdir

    # for file in listdir("data_chi_huong/grid/base_station_(250.0, 250.0)/"):
    #   if file[:4] != "gr25":
    #     continue
    #   print(file)
    #   path_sensor = "data_chi_huong/grid/base_station_(250.0, 250.0)/"+file
    #   framework = Framework(path_wce= path_wce,path_sensor= path_sensor)
    #   framework.solve()
    #   if framework.sit_flag==1:
    #     print("-------------------situation 1:")
    #     print(path_sensor)
          
