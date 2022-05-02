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
paths = ["supervised/graph_data/musicid_scen3_supervised.npz",
          "da/graph_data/musicid_scen3_DA.npz",
          "transfer/graph_data/musicid_scen3_transfer.npz",
          "multi_task/graph_data/musicid_scen3_multi task.npz",
          "simsiam/graph_data/musicid_scen3_simsiam.npz"]
          
names = ["supervised",
          "data augmentations",
          "transfer learning",
          "multi task learning",
          "simsiam"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen3_MusicID"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario 3 vs scenario 1 for simsiam and multi task
paths = ["multi_task/graph_data/musicid_scen3_multi task.npz",
          "multi_task/graph_data/musicid_scen1_multi task.npz",
          "simsiam/graph_data/musicid_scen3_simsiam.npz",
          "simsiam/graph_data/musicid_scen1_simsiam.npz"]
          
names = ["multi task learning_scen 3",
          "multi task learning_scen 1",
          "simsiam_scen 3",
          "simsiam_scen 1"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen3 vs scen1 _MusicID"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario 3 of all tasks vs scenario 1 of simsiam and multi task
paths = ["supervised/graph_data/musicid_scen3_supervised.npz",
          "da/graph_data/musicid_scen3_DA.npz",
          "transfer/graph_data/musicid_scen3_transfer.npz",
          "multi_task/graph_data/musicid_scen3_multi task.npz",
          "multi_task/graph_data/musicid_scen1_multi task.npz",
          "simsiam/graph_data/musicid_scen3_simsiam.npz",
          "simsiam/graph_data/musicid_scen1_simsiam.npz"]
          
names = ["supervised",
          "data augmentations",
          "transfer learning",
          "multi task learning_scen 3",
          "multi task learning_scen 1",
          "simsiam_scen 3",
          "simsiam_scen 1"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen3 all vs scen1 mtssl and ss_MusicID"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario transfer learn fine tuning
paths = ["transfer/graph_data/musicid_scen3_transfer.npz",
          "transfer/graph_data/musicid_scen3_ft1_transfer.npz",
          "transfer/graph_data/musicid_scen3_ft2_transfer.npz",
          "transfer/graph_data/musicid_scen3_ft3_transfer.npz",
          "transfer/graph_data/musicid_scen3_ft4_transfer.npz",
          "transfer/graph_data/musicid_scen3_ft5_transfer.npz"]
          
names = ["without fine tune",
          "1 layer fine tune",
          "2 layer fine tune",
          "3 layer fine tune",
          "4 layer fine tune",
          "all layer fine tune"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen3 transfer learn fine tuning"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario multi task learn fine tuning
paths = ["multi_task/graph_data/musicid_scen3_multi task.npz",
          "multi_task/graph_data/musicid_scen3_ft1_multi task.npz",
          "multi_task/graph_data/musicid_scen3_ft2_multi task.npz",
          "multi_task/graph_data/musicid_scen3_ft3_multi task.npz",
          "multi_task/graph_data/musicid_scen3_ft4_multi task.npz",
          "multi_task/graph_data/musicid_scen3_ft5_multi task.npz"]
          
names = ["without fine tune",
          "1 layer fine tune",
          "2 layer fine tune",
          "3 layer fine tune",
          "4 layer fine tune",
          "all layer fine tune"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen3 multi task learn fine tuning"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario simsiam learn fine tuning
paths = ["simsiam/graph_data/musicid_scen3_simsiam.npz",
          "simsiam/graph_data/musicid_scen3_ft1_simsiam.npz",
          "simsiam/graph_data/musicid_scen3_ft2_simsiam.npz",
          "simsiam/graph_data/musicid_scen3_ft3_simsiam.npz",
          "simsiam/graph_data/musicid_scen3_ft4_simsiam.npz",
          "simsiam/graph_data/musicid_scen3_ft5_simsiam.npz"]
          
names = ["without fine tune",
          "1 layer fine tune",
          "2 layer fine tune",
          "3 layer fine tune",
          "4 layer fine tune",
          "all layer fine tune"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen3 simsiam fine tuning"

plotter(paths, names, variable, variable_name, graph_name)

# Plots for scenario 3 with fine tuning
paths = ["supervised/graph_data/musicid_scen3_supervised.npz",
          "da/graph_data/musicid_scen3_DA.npz",
          "transfer/graph_data/musicid_scen3_ft5_transfer.npz",
          "multi_task/graph_data/musicid_scen3_multi task.npz",
          "simsiam/graph_data/musicid_scen3_ft5_simsiam.npz"]
          
names = ["supervised",
          "data augmentations",
          "transfer learning with fine tuning",
          "multi task learning",
          "simsiam"]
          
variable = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32,36,40,45,50,55,60]
variable_name = "samples per user"
graph_name = "kappa_scen3_MusicID_with fine tune"
plotter(paths, names, variable, variable_name, graph_name)