from trainers import *
from pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt

transformation_list=[np.array([DA_Jitter]),
                    np.array([DA_Scaling]),
                    np.array([DA_MagWarp]),
                    np.array([DA_RandSampling]),
                    np.array([DA_Flip]),
                    np.array([DA_Drop]),
                    np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop])]

sigma_lst=[np.array([0.1]),
          np.array([0.2]),
          np.array([0.2]),
          np.array([None]),
          np.array([None]),
          np.array([3]),
          np.array([0.1, 0.2, 0.2, None, None, 3])]
name_list=["Noise", "Scaling", "Magnitude Warp", "Random Sampling", "Flip", "Drop", "all"]

model_name="musicid_aug_comp_multi task"

variable_name="transformations"

acc=[]
kappa=[]
for transformations, sigma_l, name in zip(transformation_list, sigma_lst, name_list):
  acc_temp=[]
  kappa_temp=[]
  for itr in range(10):
    fet_extrct = pre_trainer(transformations, sigma_l, name)
    test_acc, kappa_score = trainer(60, fet_extrct, ft=0)
    acc_temp.append(test_acc)
    kappa_temp.append(kappa_score)
  acc.append(acc_temp)
  kappa.append(kappa_temp)
acc = np.array(acc)
kappa = np.array(kappa)
names = np.array(name_list, dtype=object)

np.savez("graph_data/"+model_name+".npz",names=names, test_acc=acc, kappa_score=kappa)
print(acc.shape)
print(kappa.shape)