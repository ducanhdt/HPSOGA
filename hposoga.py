import numpy as np
import random
import pandas as pd
from ultis import *

class Individual():
  
  def __init__(self,path):
    self.path = np.array(path)
    self.pbest = 0
    self.fitness = 0
    self.length = len(path)
    self.vec = [0 for _ in range(self.length)]
    self.path_to_position()
    
  def path_to_position(self):
    path =self.path
    position = []
    l=len(path)
    # path = np.array(path)
    for i in range(len(path)):
      x = path.index(i+1)
      print(i,x,x-l+1,path[x-l+1])
      position.append(path[x-l+1])
    self.position = position
    return position
    
  def position_to_path(self):
    path = [self.path[0]]
    for i in range(len(self.path)-1):
      path.append(self.position[path[-1]-1])
    self.path = path
    return path

  def update_pbest(self,new_value):
    if new_value > self.pbest:
      self.pbest = new_value
  def update_path(self,new_path):
    self.path = new_path
  def set_fitness(self,fitness):
    self.fitness = fitness
  
class HPSOGA():
  """
   implementation of HPSOGA
  """
  def __init__(self,init_instance,population_size,stop_condition):
    self.init_instance = init_instance
    self.population_size = population_size
    self.population = self.init_population(population_size)
    # for i in self.population:
    #   print(i.path)

  def init_population(self,number_instance):
    pop_bag = []
    for i in range(number_instance):
      rnd_sol = self.init_instance.copy()
      random.shuffle(rnd_sol)
      pop_bag.append(Individual(rnd_sol))
    return pop_bag
  
  def finness_value(self,path):
    '''
    caculate fitness value of an individual/path
    '''
    pass

  def caculate_fitness(self,):
    '''
    caculate fitness value of population
    '''
    pass

  @staticmethod
  def crossover_operation(father,mother):
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

  def crossover(self):
    '''
    crossover to get new population
    '''
    pass

  @staticmethod
  def mutation(indi, c1, c2, T, gbest):
      child = Individual(indi.path)
      sub_pbest = sub_operate(indi.pbest, indi.position)
      mul_pbest = mul_operate(sub_pbest, c1)
      sub_gbest = sub_operate(gbest, indi.position)
      mul_gbest = mul_operate(sub_gbest, c2)
      child.vec = add_operate(mul_pbest, mul_gbest)
      child.position = add_operate_x(child.vec,child.position)
      child.position_to_path()
      child.set_fitness()
      child.update_pbest(indi.pbest)
      return child

if __name__ == "__main__":

    # hposga = HPSOGA([i for i in range(0,9)],6,1)
    # print(hposga.population[0].path)
    # print(hposga.population[1].path)
    # print(hposga.crossover_operation(hposga.population[0],hposga.population[1]))
    path_to_vec([6,3,5,1,4,2])