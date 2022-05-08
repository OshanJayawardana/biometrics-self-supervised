from trainers import *
from pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from transformations import *

iters = 10

transformation_list=[[DA_Jitter],
                    [DA_Scaling],
                    [DA_MagWarp],
                    [DA_RandSampling],
                    [DA_Flip],
                    [DA_Drop],
                    [DA_TimeWarp],
                    [DA_Permutation],
                    [DA_Negation],
                    [DA_ChannelShuffle],
                    [DA_Jitter, DA_Scaling],
                    [DA_MagWarp, DA_Jitter],
                    [DA_Scaling, DA_Flip]]

#transformation_list=[[DA_Jitter],
#                    [DA_Scaling]]                    

name_list=["Noise", "Scaling", "Magnitude Warp", "Random Sampling", "Flip", "Drop", "Time Warp", "Permutation", "Negation", "Channel_Shuffle", "Noise+Scaling", "Mag_warp+Noise", "scaling+flip"]
#name_list=["Noise", "Scaling"]

model_name="musicid_aug_comp_simsiam"

variable_name="transformations"

acc_1=[]
acc_3=[]
kappa_1=[]
kappa_3=[]
name_lst_0=[]
name_lst_1=[]
name_lst_2=[]
for transformations1, name1 in zip(transformation_list, name_list):
  for transformations2, name2 in zip(transformation_list, name_list):
    acc_temp_1=[]
    kappa_temp_1=[]
    acc_temp_3=[]
    kappa_temp_3=[]
    name_lst_1.append(name1)
    name_lst_2.append(name2)
    name_lst_0.append(name1+'_'+name2)
    fet_extrct = pre_trainer(transformations1, transformations2)
    for itr in range(iters):
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
names_0 = np.array([name_lst_0], dtype=object)
names_1 = np.array([name_lst_1], dtype=object)
names_2 = np.array([name_lst_2], dtype=object)
kappa_mean = np.mean(kappa, axis=1)
kappa_mean = np.reshape(kappa_mean, (kappa_mean.shape[0],1))
kappa_csv = np.concatenate((names_0.T, names_1.T, names_2.T, kappa, kappa_mean), axis=1)

headers=["augmentation_couple", "augmentation_1", "augmentation_2"]
for itr in range(iters):
  headers.append("iter_"+str(itr))
headers.append("average")
pd.DataFrame(kappa_csv).to_csv("graph_data/"+model_name+"_scen_1_kappa.csv", index=False, header=headers)

names = np.array([name_list], dtype=object)
kappa_mean = np.mean(kappa, axis=1)
kappa_mean = np.reshape(kappa_mean, (len(name_list),len(name_list)))
kappa_matrix = np.concatenate((names.T,kappa_mean), axis=1)
pd.DataFrame(kappa_matrix).to_csv("graph_data/"+model_name+"_scen_1_kappa_matrix.csv", index=False, header=["augmentation"]+name_list)


np.savez("graph_data/"+model_name+"_scen_1.npz", names_0=names_0, names_1=names_1, names_2=names_2, test_acc=acc, kappa_score=kappa)

acc = np.array(acc_3)
kappa = np.array(kappa_3)
names_0 = np.array([name_lst_0], dtype=object)
names_1 = np.array([name_lst_1], dtype=object)
names_2 = np.array([name_lst_2], dtype=object)
kappa_mean = np.mean(kappa, axis=1)
kappa_mean = np.reshape(kappa_mean, (kappa_mean.shape[0],1))
kappa_csv = np.concatenate((names_0.T, names_1.T, names_2.T, kappa, kappa_mean), axis=1)

headers=["augmentation_couple", "augmentation_1", "augmentation_2"]
for itr in range(iters):
  headers.append("iter_"+str(itr))
headers.append("average")
pd.DataFrame(kappa_csv).to_csv("graph_data/"+model_name+"_scen_3_kappa.csv", index=False, header=headers)

names = np.array([name_list], dtype=object)
kappa_mean = np.mean(kappa, axis=1)
kappa_mean = np.reshape(kappa_mean, (len(name_list),len(name_list)))
kappa_matrix = np.concatenate((names.T,kappa_mean), axis=1)
pd.DataFrame(kappa_matrix).to_csv("graph_data/"+model_name+"_scen_3_kappa_matrix.csv", index=False, header=["augmentation"]+name_list)

np.savez("graph_data/"+model_name+"_scen_3.npz", names_0=names_0, names_1=names_1, names_2=names_2, test_acc=acc, kappa_score=kappa)