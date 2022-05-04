import numpy as np
import matplotlib.pyplot as plt

def plotter(paths, names, variables, variable_name, graph_name):
  kappa = []
  for path in paths:
    data = np.load(path)
    score = data["kappa_score"]
    cal_score = np.mean(score, axis=1)
    kappa.append(cal_score)
  
  plt.figure(figsize=(12,8))
  for i in range(len(kappa)):
    plt.plot(variable, kappa[i], label=names[i])
  plt.title("kappa score vs "+variable_name)
  plt.xlabel(variable_name)
  plt.ylabel("kappa score")
  plt.legend()
  plt.show()
  plt.savefig('graphs/'+graph_name+'.jpg')
  plt.close()
  return True

# Plots for scenario 3 without fine tuning
paths = ["multi_task/graph_data/musicid_scen2_multi task.npz",
          "simsiam/graph_data/musicid_scen2_simsiam.npz",
          "transfer/graph_data/musicid_scen2_transfer.npz",
          "transfer_da/graph_data/musicid_scen2_transfer_da.npz"]
          
names = ["multi task learning",
          "simsiam",
          "transfer learning",
          "transfer learning da"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen2_MusicID"

plotter(paths, names, variable, variable_name, graph_name)