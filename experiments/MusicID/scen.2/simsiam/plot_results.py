from trainers import *
from pre_trainers import *
from cl_pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists

fet_extrct_sim = pre_trainer()

variable_name="sampes per user"
model_name="musicid_scen2_simsiam"
variable=[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
#variable=[1]
acc=[]
kappa=[]
el95 = 0
for el in variable:
  for itr in range(10):
    test_acc, kappa_score, model = cl_pre_trainer(el, fet_extrct_sim)
    if test_acc>=0.95:
      el95 = el
      break
  if el95!=0:
    break
else:
  el95 = el

model.summary()
acc=[]
kappa=[]
for el in variable:
  acc_temp=[]
  kappa_temp=[]
  for itr in range(10):
    test_acc, kappa_score = trainer(samples_per_user=el, fet_extrct=model, ft=5)
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

print("samples per user at 95% accuracy: ", el95)
if not exists("samples_per_user_pretraining.txt"):
  f = open("samples_per_user_pretraining.txt", "x")

f = open("samples_per_user_pretraining.txt", "w")
f.write(str(el95))
f.close()