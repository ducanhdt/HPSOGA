import numpy as np
import random
import pandas as pd

def cosine_distances(v1,v2):
  return np.linalg.norm(v1-v2)

def path_to_vec(path):
  vec = []
  l=len(path)
  # path = np.array(path)
  for i in range(len(path)):
    x = path.index(i+1)
    print(i,x,x-l+1,path[x-l+1])
    vec.append(path[x-l+1])
  return vec

# def vec_to_path(vec):


def add_operate(v2, v1):
    v = [0 for _ in v2]
    for i, _ in enumerate(v2):
        if v2[i] == 0:
            v[i] = v1[i]
        else:
            v[i] = v2[i]
    return v


def mul_operate(v1, c):
    v = [0 for _ in v1]
    r = [random.random() for _ in v1]
    for i, _ in enumerate(v):
        if r[i] < c:
          v[i] = v1[i]
    return v


def sub_operate(x2, x1):
    v = [0 for _ in x2]
    for i, _ in enumerate(x2):
        if x2[i] != x1[i]:
            v[i] = x2[i]
    return v

def add_operate_x(v,x):
    res = x.copy()
    for i in range(len(v)):
      if v[i]!=0 and v[i]!=x[i]:
        res[res.index(v[i])] = res[v[i]-1]
        res[v[i]-1] = res[i]
        res[i] = v[i]
    return res
def checksum(path):
  return len(list(set(path)))==len(path)

if __name__ == "__main__":
  x = [14, 7, 16, 12, 3, 1, 5, 2, 15, 20, 10, 11, 13, 9, 19, 8, 6, 17, 18, 4]
  v = [16, 7, 13, 10, 18, 1, 6, 2, 15, 0, 0, 0, 0, 0, 12, 19, 14, 11, 3, 5]
  print(x)
  print(v)
  n = add_operate_x(v,x)
  print(checksum(n))
  print(n)
  print(sorted([1,3,2],reverse=True))