from trainers import *
from pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

iters = 10
con = 2

transformation_list=[np.array([DA_Jitter]),
                    np.array([DA_Scaling]),
                    np.array([DA_MagWarp]),
                    np.array([DA_RandSampling]),
                    np.array([DA_Flip]),
                    np.array([DA_Drop]),
                    np.array([DA_TimeWarp]),
                    np.array([DA_Negation]),
                    np.array([DA_ChannelShuffle]),
                    np.array([DA_Permutation]),
                    np.array([DA_Jitter, DA_Scaling]),
                    np.array([DA_MagWarp, DA_Drop]),
                    np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_Drop]),
                    np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop]),
                    np.array([DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop, DA_TimeWarp, DA_Negation, DA_ChannelShuffle, DA_Permutation])]
                    
#transformation_list=[np.array([DA_Jitter]),
#                    np.array([DA_Scaling])]

sigma_lst=[np.array([0.1*con]), #Noise
          np.array([0.2*con]), #Scaling
          np.array([0.2*con]), #Magnitude Warp
          np.array([None]), #Random Sampling
          np.array([None]), #Flip
          np.array([3]), #Drop
          np.array([0.2*con]), #TimeWarp
          np.array([None]), #Negation
          np.array([None]), #Channel shuffle
          np.array([0.1*con]), #Permutations
          np.array([0.1*con,0.2*con]), #Noise,scale
          np.array([0.2*con,3]), #Magnitude Warp, Drop
          np.array([0.1*con,0.2*con,0.2*con,3]), #Noise,scale,Magnitude Warp, Drop
          np.array([0.1*con, 0.2*con, 0.2*con, None, None, 3]), #all1 DA_Jitter, DA_Scaling, DA_MagWarp, DA_RandSampling, DA_Flip, DA_Drop
          np.array([0.1*con, 0.2*con, 0.2*con, None, None, 3, 0.2*con, None, None, 0.1*con])] #all
#sigma_lst=[np.array([0.1]),
#          np.array([0.2])]

name_list=["Noise", "Scaling", "Magnitude Warp", "Random Sampling", "Flip", "Drop",
          "TimeWarp",
          "Negation",
          "Channel shuffle",
          "Permutations",
          "Noise,scale", "Magnitude Warp, Drop", "Noise,scale,Magnitude Warp, Drop","all_0", "all_1"]
#name_list=["Noise", "Scaling"]

model_name="musicid_aug_comp_multi task"

variable_name="transformations"

acc_1=[]
acc_3=[]
kappa_1=[]
kappa_3=[]
for transformations, sigma_l, name in zip(transformation_list, sigma_lst, name_list):
  acc_temp_1=[]
  kappa_temp_1=[]
  acc_temp_3=[]
  kappa_temp_3=[]
  for itr in range(iters):
    fet_extrct = pre_trainer(transformations, sigma_l, name)
    test_acc_1, kappa_score_1 = trainer(60, fet_extrct,scen=1, ft=0)
    test_acc_3, kappa_score_3 = trainer(60, fet_extrct,scen=3, ft=0)
    acc_temp_1.append(test_acc_1)
    acc_temp_3.append(test_acc_3)
    kappa_temp_1.append(kappa_score_1)
    kappa_temp_3.append(kappa_score_3)
  acc_1.append(acc_temp_1)
  acc_3.append(acc_temp_3)
  kappa_1.append(kappa_temp_1)
  kappa_3.append(kappa_temp_3)
  
acc = np.array(acc_1)
kappa = np.array(kappa_1)
names = np.array([name_list], dtype=object)
kappa_mean = np.mean(kappa, axis=1)
kappa_mean = np.reshape(kappa_mean, (kappa_mean.shape[0],1))
print("names shape", names.T.shape)
print("data shape", kappa.shape)
print("kappa mean shape", kappa_mean.shape)
kappa_csv = np.concatenate((names.T, kappa, kappa_mean), axis=1)

headers=["augmentation"]
for itr in range(iters):
  headers.append("iter_"+str(itr))
headers.append("average")
pd.DataFrame(kappa_csv).to_csv("graph_data/"+model_name+"_scen_1_kappa.csv", index=False, header=headers)

np.savez("graph_data/"+model_name+"_scen_1.npz",names=names, test_acc=acc, kappa_score=kappa)

acc = np.array(acc_3)
kappa = np.array(kappa_3)
names = np.array([name_list], dtype=object)
kappa_mean = np.mean(kappa, axis=1)
kappa_mean = np.reshape(kappa_mean, (kappa_mean.shape[0],1))
kappa_csv = np.concatenate((names.T, kappa, kappa_mean), axis=1)

headers=["augmentation"]
for itr in range(iters):
  headers.append("iter_"+str(itr))
headers.append("average")
pd.DataFrame(kappa_csv).to_csv("graph_data/"+model_name+"_scen_3_kappa.csv", index=False, header=headers)

np.savez("graph_data/"+model_name+"_scen_3.npz",names=names, test_acc=acc, kappa_score=kappa)