from trainers import *
from pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

iters = 10
property = "configuration"

model_name="musicid_config_comp_simsiam"

name_list = ["16", "16, 32", "48, 96", "64, 128", "16, 32, 64", "48, 96, 192"]

acc_1=[]
acc_3=[]
kappa_1=[]
kappa_3=[]
for config_num in range(6):
  acc_temp_1=[]
  kappa_temp_1=[]
  acc_temp_3=[]
  kappa_temp_3=[]
  fet_extrct = pre_trainer(config_num)
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
names = np.array([name_list], dtype=object)
kappa_mean = np.mean(kappa, axis=1)
kappa_mean = np.reshape(kappa_mean, (kappa_mean.shape[0],1))
print("names shape", names.T.shape)
print("data shape", kappa.shape)
print("kappa mean shape", kappa_mean.shape)
kappa_csv = np.concatenate((names.T, kappa, kappa_mean), axis=1)

headers=[property]
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

headers=[property]
for itr in range(iters):
  headers.append("iter_"+str(itr))
headers.append("average")
pd.DataFrame(kappa_csv).to_csv("graph_data/"+model_name+"_scen_3_kappa.csv", index=False, header=headers)

np.savez("graph_data/"+model_name+"_scen_3.npz",names=names, test_acc=acc, kappa_score=kappa)