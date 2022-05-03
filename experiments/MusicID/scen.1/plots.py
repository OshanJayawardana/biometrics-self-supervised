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

# Plots for scenario 1 without fine tuning
paths = ["supervised/graph_data/musicid_scen1_supervised.npz",
          "da/graph_data/musicid_scen1_DA.npz",
          "multi_task/graph_data/musicid_scen1_multi task.npz",
          "simsiam/graph_data/musicid_scen1_simsiam.npz"]
          
names = ["supervised",
          "data augmentations",
          "multi task learning",
          "simsiam"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen1_MusicID"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario multi task learn fine tuning
paths = ["multi_task/graph_data/musicid_scen1_multi task.npz",
          "multi_task/graph_data/musicid_scen1_ft1_multi task.npz",
          "multi_task/graph_data/musicid_scen1_ft2_multi task.npz",
          "multi_task/graph_data/musicid_scen1_ft3_multi task.npz",
          "multi_task/graph_data/musicid_scen1_ft4_multi task.npz",
          "multi_task/graph_data/musicid_scen1_ft5_multi task.npz"]
          
names = ["without fine tune",
          "1 layer fine tune",
          "2 layer fine tune",
          "3 layer fine tune",
          "4 layer fine tune",
          "all layer fine tune"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen1 multi task learn fine tuning"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario simsiam learn fine tuning
paths = ["simsiam/graph_data/musicid_scen1_simsiam.npz",
          "simsiam/graph_data/musicid_scen1_ft1_simsiam.npz",
          "simsiam/graph_data/musicid_scen1_ft2_simsiam.npz",
          "simsiam/graph_data/musicid_scen1_ft3_simsiam.npz",
          "simsiam/graph_data/musicid_scen1_ft4_simsiam.npz",
          "simsiam/graph_data/musicid_scen1_ft5_simsiam.npz"]
          
names = ["without fine tune",
          "1 layer fine tune",
          "2 layer fine tune",
          "3 layer fine tune",
          "4 layer fine tune",
          "all layer fine tune"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen1 simsiam fine tuning"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario 3 with fine tuning
paths = ["supervised/graph_data/musicid_scen1_supervised.npz",
          "da/graph_data/musicid_scen1_DA.npz",
          "multi_task/graph_data/musicid_scen1_ft5_multi task.npz",
          "simsiam/graph_data/musicid_scen1_ft5_simsiam.npz"]
          
names = ["supervised",
          "data augmentations",
          "multi task learning with fine tune",
          "simsiam with fine tune"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen1_MusicID_with fine tune"

plotter(paths, names, variable, variable_name, graph_name)