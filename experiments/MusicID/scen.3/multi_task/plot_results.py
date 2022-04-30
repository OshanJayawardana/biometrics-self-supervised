from trainers import *
import numpy as np
import matplotlib.pyplot as plt

variable_name="samples per user"
model_name="musicid_scen3_multi task"
variable=[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable=[15,20]
acc=[]
kappa=[]
for el in variable:
  acc_temp=[]
  kappa_temp=[]
  for itr in range(2):
    test_acc, kappa_score = trainer(el)
    acc_temp.append(test_acc)
    kappa_temp.append(kappa_score)
  acc.append(acc_temp)
  kappa.append(kappa_temp)
acc = np.array(acc)
kappa = np.array(kappa)

np.savez("graph_data/"+model_name+".npz", test_acc=acc, kappa_score=kappa)
print(acc.shape)
print(kappa.shape)

kappa_max = np.max(kappa, axis=1)
plt.figure(figsize=(12,8))
plt.plot(variable,kappa_max, 'm', label=model_name)
plt.title("kappa score vs "+variable_name)
plt.xlabel(variable_name)
plt.ylabel("kappa score")
plt.legend()
plt.show()
plt.savefig('graphs/kappa.jpg')
plt.close()

acc_max = np.max(acc, axis=1)
plt.figure(figsize=(12,8))
plt.plot(variable,acc_max, 'm', label=model_name)
plt.title("test accuracy vs "+variable_name)
plt.xlabel(variable_name)
plt.ylabel("test acuracy")
plt.legend()
plt.show()
plt.savefig('graphs/acc.jpg')
plt.close()