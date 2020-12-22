import numpy as np
import random
import pandas as pd
# x =[9,8,4,5,6,7,1,3,2,0]
# a= x.copy()
# random.shuffle(a)
# mother = {'path':a}
# a= x.copy()
# random.shuffle(a)
# father = {'path':a}
# start,end = np.sort(np.random.choice(len(mother['path']),2,replace=False))


# def vec_to_path(sepath,vec):
#   path = [sepath[0]]
#   for i in range(len(sepath)-1):
#     print(path[-1])
#     path.append(vec[path[-1]-1])
#   return path

# print(vec_to_path([6,1,2,3,4,5],[4,6,5,2,1,3]))
import os 
base_path = "data_chi_huong/uniform/base_station_(250.0, 250.0)"
for i in os.listdir(base_path):
  print(base_path+'/'+i)