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

paths = ["supervised/graph_data/musicid_scen3_supervised.npz",
          "da/graph_data/musicid_scen3_DA.npz",
          "transfer/graph_data/musicid_scen3_supervised.npz",
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

paths = ["supervised/graph_data/musicid_scen3_supervised.npz",
          "da/graph_data/musicid_scen3_DA.npz",
          "transfer/graph_data/musicid_scen3_supervised.npz",
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