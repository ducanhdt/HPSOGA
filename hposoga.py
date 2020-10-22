import numpy as np
import random
import pandas as pd
from ultis import *

class Individual():
  def __init__(self,path):
    self.path = np.array(path)
    self.pbest = 0
  def update_pbest(self,new_value):
    if new_value > self.pbest:
      self.pbest = new_value
  def update_path(self,new_path):
    self.path = new_path

  
class HPSOGA():
  """
   implementation of HPSOGA
  """
  def __init__(self,init_instance,population_size,stop_condition):
    self.init_instance = init_instance
    self.population_size = population_size
    self.population = self.init_population(population_size)
    for i in self.population:
      print(i.path)

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

  def crossover(self,):
    '''
    crossover to get new population
    '''
    pass

  def mutation(self,):
    '''
    PSO mutation
    '''
    pass

if __name__ == "__main__":
    hposga = HPSOGA([1,2,3,4],6,1)